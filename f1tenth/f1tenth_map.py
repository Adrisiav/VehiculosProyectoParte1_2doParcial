import os
import cv2
import csv
import yaml
import numpy as np
from pathlib import Path

import math
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning.utils import Grid, SearchFactory  # usa tu SearchFactory existente

# ================= Utilidades de mapa =================

def load_map(yaml_path, downsample_factor=1, return_gray_for_viz=True):
    with open(yaml_path, 'r') as f:
        map_config = yaml.safe_load(f)
    img_path = map_config['image']
    map_img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if map_img_gray is None:
        raise FileNotFoundError(f"No pude leer la imagen del mapa: {img_path}")

    resolution = float(map_config['resolution'])
    origin = map_config['origin']  # [x0, y0, yaw]

    # Binarizar: 1 = ocupado (pared), 0 = libre
    map_bin = np.zeros_like(map_img_gray, dtype=np.uint8)
    map_bin[map_img_gray < int(0.45 * 255)] = 1

    # Engrosar paredes para robustez
    if downsample_factor > 12:
        map_bin = cv2.dilate(map_bin, np.ones((5, 5), np.uint8), iterations=2)
    elif downsample_factor >= 4:
        map_bin = cv2.dilate(map_bin, np.ones((3, 3), np.uint8), iterations=1)

    # Downsample
    map_bin = map_bin.astype(np.float32)
    h, w = map_bin.shape
    new_h, new_w = h // downsample_factor, w // downsample_factor
    map_bin = cv2.resize(map_bin, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Re-binarizar
    if downsample_factor > 12:
        map_bin = (map_bin > 0.10).astype(np.uint8)
    elif downsample_factor >= 4:
        map_bin = (map_bin > 0.25).astype(np.uint8)
    else:
        map_bin = (map_bin >= 0.5).astype(np.uint8)

    resolution_eff = resolution * downsample_factor
    map_img_gray_ds = None
    if return_gray_for_viz:
        map_img_gray_ds = cv2.resize(map_img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return map_bin, resolution_eff, origin, map_img_gray_ds


def world_to_map(x_world, y_world, resolution, origin):
    x_map = int((x_world - origin[0]) / resolution)
    y_map = int((y_world - origin[1]) / resolution)
    return (x_map, y_map)


def map_to_world(x_map, y_map, resolution, origin):
    x_world = x_map * resolution + origin[0]
    y_world = y_map * resolution + origin[1]
    return (x_world, y_world)


def to_world_path(path, resolution, origin):
    out = []
    for (x, y) in path or []:
        if isinstance(x, int) and isinstance(y, int):
            out.append(map_to_world(x, y, resolution, origin))
        else:
            out.append((float(x), float(y)))
    return out

# ================= Entorno Grid (Theta*) =================

def grid_from_map(map_bin):
    h, w = map_bin.shape
    env = Grid(w, h)
    obstacles = {(x, h - 1 - y) for y in range(h) for x in range(w) if map_bin[y, x] == 1}
    env.update(obstacles)
    return env

# ================= Re-muestreo & suavizado =================

def resample_by_spacing(points, spacing_m):
    if not points or len(points) < 2:
        return points
    pts = np.array(points, dtype=float)
    dif = np.diff(pts, axis=0)
    seglen = np.linalg.norm(dif, axis=1)
    L = np.concatenate([[0.0], np.cumsum(seglen)])
    total = L[-1]
    if total <= 1e-9:
        return [tuple(pts[0])]
    n_new = int(np.floor(total / spacing_m)) + 1
    s = np.linspace(0.0, total, n_new)
    x = np.interp(s, L, pts[:, 0])
    y = np.interp(s, L, pts[:, 1])
    out = np.stack([x, y], axis=1)
    if np.linalg.norm(out[-1] - pts[-1]) > 1e-6:
        out = np.vstack([out, pts[-1]])
    return [tuple(p) for p in out]


def chaikin_smooth(points, iters=3, keep_ends=True):
    if len(points) < 3 or iters <= 0:
        return points
    pts = [np.array(p, dtype=float) for p in points]
    for _ in range(iters):
        new_pts = []
        if keep_ends:
            new_pts.append(pts[0])
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            Q = 0.75 * p + 0.25 * q
            R = 0.25 * p + 0.75 * q
            new_pts.extend([Q, R])
        if keep_ends:
            new_pts.append(pts[-1])
        pts = new_pts
    return [tuple(p) for p in pts]


def bspline_smooth(points, num=None, degree=3):
    try:
        from scipy.interpolate import splprep, splev
    except Exception:
        if num is None:
            num = max(100, len(points) * 3)
        return chaikin_smooth(points, iters=3), False
    if len(points) < degree + 1:
        return points, True
    pts = np.array(points, dtype=float).T
    tck, _u = splprep(pts, s=0.0, k=min(degree, len(points)-1))
    if num is None:
        num = max(100, len(points) * 3)
    unew = np.linspace(0, 1, num)
    out = splev(unew, tck)
    smoothed = list(zip(out[0], out[1]))
    return smoothed, True

# ================= Guardado / Visualización =================

def save_csv(points, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for x, y in points or []:
            w.writerow([x, y])

def draw_on_bin(map_bin, path_world, origin, resolution, filename):
    img = cv2.cvtColor(((1 - map_bin) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    h = map_bin.shape[0]
    def world_to_px(p):
        x, y = p
        px = int((x - origin[0]) / resolution)
        py = h - 1 - int((y - origin[1]) / resolution)
        return px, py
    if path_world:
        for p in path_world:
            px, py = world_to_px(p)
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)
    cv2.imwrite(filename, img)

# ================= DWA (Dynamic Window Approach) =================

def collision_free(map_bin, resolution, origin, x, y, robot_radius=0.2):
    """Chequeo simple por ‘disco’: si cualquier pixel del disco cae en pared, colisión."""
    h, w = map_bin.shape
    xm = int((x - origin[0]) / resolution)
    ym = h - 1 - int((y - origin[1]) / resolution)
    if xm < 0 or ym < 0 or xm >= w or ym >= h:
        return False
    r_pix = max(1, int(robot_radius / resolution))
    x0, x1 = max(0, xm - r_pix), min(w - 1, xm + r_pix)
    y0, y1 = max(0, ym - r_pix), min(h - 1, ym + r_pix)
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            if (xx - xm)**2 + (yy - ym)**2 <= r_pix**2:
                if map_bin[yy, xx] == 1:
                    return False
    return True

def simulate_unicycle(x, y, yaw, v, w, dt):
    nx = x + v * math.cos(yaw) * dt
    ny = y + v * math.sin(yaw) * dt
    nyaw = yaw + w * dt
    # Normaliza ángulo a [-pi, pi]
    nyaw = (nyaw + math.pi) % (2 * math.pi) - math.pi
    return nx, ny, nyaw

def nearest_point_index(path, x, y):
    d2min, idx = float('inf'), 0
    for i, (px, py) in enumerate(path):
        d2 = (px - x)**2 + (py - y)**2
        if d2 < d2min:
            d2min, idx = d2, i
    return idx

def target_point(path, x, y, lookahead=0.8):
    """‘pure pursuit’ simple: objetivo a cierta distancia por delante en la ruta."""
    if not path:
        return None, len(path)-1
    i0 = nearest_point_index(path, x, y)
    acc = 0.0
    for i in range(i0, len(path)-1):
        seg = math.dist(path[i], path[i+1])
        acc += seg
        if acc >= lookahead:
            return path[i+1], i+1
    return path[-1], len(path)-1

def dwa_control_step(state, path, map_bin, resolution, origin, params):
    """
    state = (x, y, yaw, v, w)
    params: dict con:
      v_max, v_min, w_max, acc_v, acc_w, dt, horizon_T,
      weight_goal, weight_clearance, weight_speed,
      robot_radius, lookahead
    """
    x, y, yaw, v0, w0 = state
    dt = params["dt"]
    T = params["horizon_T"]
    v_max = params["v_max"]; v_min = params["v_min"]
    w_max = params["w_max"]
    acc_v = params["acc_v"]; acc_w = params["acc_w"]

    # Dynamic window
    v_low  = max(v_min, v0 - acc_v * dt)
    v_high = min(v_max, v0 + acc_v * dt)
    w_low  = max(-w_max, w0 - acc_w * dt)
    w_high = min( w_max, w0 + acc_w * dt)

    best_cost = float('inf')
    best_vw = (0.0, 0.0)
    best_traj_end = (x, y, yaw)

    # Sampling
    for v in np.linspace(v_low, v_high, 6):
        for w in np.linspace(w_low, w_high, 7):
            # Simula trayectoria
            sx, sy, syaw = x, y, yaw
            ok = True
            min_clear = 1e9
            for _ in range(int(T / dt)):
                sx, sy, syaw = simulate_unicycle(sx, sy, syaw, v, w, dt)
                if not collision_free(map_bin, resolution, origin, sx, sy, params["robot_radius"]):
                    ok = False
                    break
                # clearance aproximado = círculo libre más cercano (radio en m)
                # aquí usamos pixel-visit rápido: si el centro está libre, clearance ~ robot_radius
                min_clear = min(min_clear, params["robot_radius"])
            if not ok:
                continue

            # Costos
            tgt, _ = target_point(path, sx, sy, params["lookahead"])
            if tgt is None:
                continue
            heading_err = math.atan2(tgt[1] - sy, tgt[0] - sx) - syaw
            heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
            cost_goal = abs(heading_err)  # menor es mejor

            cost_clear = 1.0 / (min_clear + 1e-3)  # preferir mayor holgura
            cost_speed = (params["v_max"] - v)     # preferir mayor velocidad

            cost = (params["weight_goal"] * cost_goal +
                    params["weight_clearance"] * cost_clear +
                    params["weight_speed"] * cost_speed)

            if cost < best_cost:
                best_cost = cost
                best_vw = (v, w)
                best_traj_end = (sx, sy, syaw)

    # Si nada fue válido (muy raro), frena
    if best_cost == float('inf'):
        return (0.0, 0.0), (x, y, yaw)
    return best_vw, best_traj_end

# ================= Pipeline principal =================

def theta_star_pipeline(map_yaml_path,
                        x_start, y_start, x_goal, y_goal,
                        downsample_factor=5,
                        outdir="results_theta_dwa"):
    # 1) Cargar mapa
    map_bin, resolution, origin, map_img_gray = load_map(map_yaml_path, downsample_factor, True)

    # 2) Theta* en grid
    env = grid_from_map(map_bin)
    sf = SearchFactory()
    start_idx = world_to_map(x_start, y_start, resolution, origin)
    goal_idx  = world_to_map(x_goal, y_goal, resolution, origin)
    planner = sf("theta_star", start=start_idx, goal=goal_idx, env=env)
    planner.run()
    cost, path, _ = planner.plan()
    path_world = to_world_path(path, resolution, origin)

    # 3) Guardar crudo y visualizar
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_csv(path_world, f"{outdir}/theta_raw.csv")
    draw_on_bin(map_bin, path_world, origin, resolution, f"{outdir}/theta_raw.png")

    # 4) Re-muestrear a 0.5 m y 1.0 m + suavizar
    outputs = {}
    for spacing in [0.5, 1.0]:
        resampled = resample_by_spacing(path_world, spacing)
        smoothed, used_bs = bspline_smooth(resampled)
        if not used_bs:
            smoothed = chaikin_smooth(resampled, iters=3)
        tag = f"{str(spacing).replace('.','p')}m"
        save_csv(resampled, f"{outdir}/theta_{tag}_resampled.csv")
        save_csv(smoothed,  f"{outdir}/theta_{tag}_smoothed.csv")
        draw_on_bin(map_bin, resampled, origin, resolution, f"{outdir}/theta_{tag}_resampled.png")
        draw_on_bin(map_bin, smoothed,  origin, resolution, f"{outdir}/theta_{tag}_smoothed.png")
        outputs[spacing] = smoothed

    # 5) DWA para seguir la ruta (elige 0.5 m o 1.0 m)
    path_to_follow = outputs[0.5]  # puedes cambiar a 1.0 según la rúbrica
    # Parámetros DWA (ajústalos a tu robot)
    params = dict(
        v_max=2.0, v_min=0.0,          # m/s
        w_max=2.5,                     # rad/s
        acc_v=1.5, acc_w=3.0,          # m/s^2, rad/s^2
        dt=0.05, horizon_T=0.8,        # paso y horizonte de predicción
        weight_goal=4.0, weight_clearance=2.0, weight_speed=0.5,
        robot_radius=0.18,             # ~ F1Tenth footprint
        lookahead=0.9
    )

    state = (x_start, y_start, 0.0, 0.0, 0.0)  # (x, y, yaw, v, w)
    traj = [(state[0], state[1])]
    vw_log = []
    max_steps = 4000
    goal_tol = 0.25

    for _ in range(max_steps):
        if not path_to_follow:
            break
        # meta alcanzada
        if math.dist((state[0], state[1]), path_to_follow[-1]) <= goal_tol:
            break
        (v_cmd, w_cmd), (nx, ny, nyaw) = dwa_control_step(state, path_to_follow, map_bin, resolution, origin, params)
        state = (nx, ny, nyaw, v_cmd, w_cmd)
        traj.append((nx, ny))
        vw_log.append((v_cmd, w_cmd))

    save_csv(traj, f"{outdir}/dwa_traj.csv")
    draw_on_bin(map_bin, traj, origin, resolution, f"{outdir}/dwa_traj.png")
    print(f"Listo. Resultados en: {outdir}")
    return {
        "raw": path_world,
        "resampled_0p5": outputs[0.5],
        "resampled_1p0": outputs[1.0],
        "dwa_traj": traj
    }

# ================= MAIN =================

if __name__ == "__main__":
    # Configura tu mapa y puntos
    map_yaml_path = "Oschersleben_map.yaml"
    downsample_factor = 5

    # Puntos en coordenadas del mundo (ajústalos a tu mapa)
    x_start, y_start = 4.0, -0.8
    x_goal, y_goal   = 1.0, -0.5

    theta_star_pipeline(
        map_yaml_path,
        x_start, y_start, x_goal, y_goal,
        downsample_factor=downsample_factor,
        outdir="results_theta_dwa"
    )
import os
import cv2
import csv
import yaml
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning.utils import Grid, Map, SearchFactory

# ---------------- Funciones ----------------

# ---------------- Funciones ----------------

def load_map(yaml_path, downsample_factor=1):
    with open(yaml_path, 'r') as f:
        map_config = yaml.safe_load(f)

    img_path = map_config['image']
    map_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resolution = map_config['resolution']
    origin = map_config['origin']

    # Binarizar: 1 = ocupado, 0 = libre
    map_bin = np.zeros_like(map_img, dtype=np.uint8)
    map_bin[map_img < int(0.45 * 255)] = 1

    # Engrosar obstáculos según el factor
    if downsample_factor > 12:
        map_bin = cv2.dilate(map_bin, np.ones((5, 5), np.uint8), iterations=2)
    elif downsample_factor >= 4:
        map_bin = cv2.dilate(map_bin, np.ones((3, 3), np.uint8), iterations=1)

    # Downsampling
    map_bin = map_bin.astype(np.float32)
    h, w = map_bin.shape
    new_h, new_w = h // downsample_factor, w // downsample_factor
    map_bin = cv2.resize(map_bin, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Re-binarizar
    if downsample_factor > 12:
        map_bin = (map_bin > 0.10).astype(np.uint8)
    elif downsample_factor >= 4:
        map_bin = (map_bin > 0.25).astype(np.uint8)
    else:
        map_bin = (map_bin >= 0.5).astype(np.uint8)

    resolution *= downsample_factor
    return map_bin, resolution, origin


def grid_from_map(map_bin, resolution, origin):
    h, w = map_bin.shape
    env = Grid(w, h)
    obstacles = {(x, h - 1 - y) for y in range(h) for x in range(w) if map_bin[y, x] == 1}
    env.update(obstacles)
    return env


def map_from_yaml(map_bin, resolution, origin):
    h, w = map_bin.shape
    env = Map(w * resolution, h * resolution)

    # No agregamos obstáculos manuales (obs_rect ni obs_circ)
    # Solo definimos el límite del mapa
    env.update(obs_rect=[], obs_circ=[])
    env.boundary = [[0, 0, w * resolution, h * resolution]]

    return env



def world_to_map(x_world, y_world, resolution, origin):
    x_map = int((x_world - origin[0]) / resolution)
    y_map = int((y_world - origin[1]) / resolution)
    return (x_map, y_map)


def map_to_world(x_map, y_map, resolution, origin):
    x_world = x_map * resolution + origin[0]
    y_world = y_map * resolution + origin[1]
    return (x_world, y_world)


def save_path_as_csv(path, filename, resolution, origin):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        if path is not None:
            for x_map, y_map in reversed(path):
                x, y = map_to_world(x_map, y_map, resolution, origin)
                writer.writerow([x, y])


def draw_path_on_map(map_bin, path, origin, resolution, filename="path_result.png"):
    img = cv2.cvtColor((map_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if path is not None:
        for point in path:
            x, y = point
            px = int((x - origin[0]) / resolution)
            py = map_bin.shape[0] - 1 - int((y - origin[1]) / resolution)
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)
    cv2.imshow("Path", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(filename, img)


# ---------------- Main ----------------

if __name__ == "__main__":

    map_yaml_path = "Oschersleben_map.yaml"
    downsample_factor = 5

    x_start, y_start = 4, -0.8
    x_goal, y_goal = 1, -0.5

    # Tipo de planificador: "a_star", "theta_star", "dijkstra", "gbfs", "rrt", "rrt_star", "rrt_connect", "informed_rrt"
    planner_type = "rrt"  # Cambia aquí para probar otros planificadores

    # Cargar mapa
    map_bin, resolution, origin = load_map(map_yaml_path, downsample_factor)

    search_factory = SearchFactory()

    # Selección de entorno y planificador
    if planner_type in ["a_star", "theta_star", "dijkstra", "gbfs"]:
        env = grid_from_map(map_bin, resolution, origin)
        start_idx = world_to_map(x_start, y_start, resolution, origin)
        goal_idx = world_to_map(x_goal, y_goal, resolution, origin)
        planner = search_factory(planner_type, start=start_idx, goal=goal_idx, env=env)
    else:
        env = map_from_yaml(map_bin, resolution, origin)
        start_pos = (x_start, y_start)
        goal_pos = (x_goal, y_goal)
        planner = search_factory(planner_type, start=start_pos, goal=goal_pos, env=env)

    # Ejecutar búsqueda
    planner.run()
    cost, path, _ = planner.plan()

    # Guardar CSV
    save_path_as_csv(path, f"{planner_type}_path_real.csv", resolution, origin)
    print(f"Ruta guardada como {planner_type}_path_real.csv")

    # Dibujar ruta
    draw_path_on_map(map_bin, path, origin, resolution, f"{planner_type}_path.png")