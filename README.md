# VehiculosProyectoParte1_2doParcial
# Planificador de Trayectorias para F1TENTH – Theta* + DWA

Este repositorio permite cargar mapas tipo YAML + PNG, realizar la binarización, aplicar reducción de resolución (downsampling), convertir a formato de cuadrícula, ejecutar el planificador Theta* y generar trayectorias suavizadas. También permite simular el seguimiento de la ruta usando el controlador DWA (Dynamic Window Approach).

---

## Requisitos

- Python 3.8 o superior  
- pip  
- python3-venv (para crear entornos virtuales)  
- Sistema operativo Linux (probado en Ubuntu 22.04)  

---

## Instalación

### 1. Crear entorno virtual
```bash
sudo apt update
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate
```

2. Instalar dependencias
```
pip install --upgrade pip
pip install -r requirements.txt
```

Uso
1. Configuración del mapa y puntos

Dentro del archivo principal (main.py o tu script), configura:
```
map_yaml_path = "Oschersleben_map.yaml"
downsample_factor = 5

x_start, y_start = 4.0, -0.8
x_goal, y_goal   = 1.0, -0.5
```
2. Ejecutar pipeline Theta* + DWA
```
python3 main.py
```

Esto ejecutará:

    Carga del mapa y binarización.

    Theta* en grid para planificar la ruta.

    Re-muestreo a distancias de 0.5 m y 1.0 m.

    Suavizado de trayectoria (Chaikin o B-spline).

    Simulación de seguimiento con DWA.

Salida

Los resultados se generan en la carpeta results_theta_dwa (puedes cambiarla en outdir):

    theta_raw.csv – Ruta cruda en coordenadas del mundo real.

    theta_0p5m_resampled.csv – Ruta re-muestreada a 0.5 m.

    theta_0p5m_smoothed.csv – Ruta suavizada a 0.5 m.

    theta_1p0m_resampled.csv – Ruta re-muestreada a 1.0 m.

    theta_1p0m_smoothed.csv – Ruta suavizada a 1.0 m.

    theta_raw.png – Visualización de ruta cruda sobre el mapa.

    theta_0p5m_resampled.png – Visualización de ruta re-muestreada 0.5 m.

    theta_0p5m_smoothed.png – Visualización de ruta suavizada 0.5 m.

    dwa_traj.csv – Trayectoria generada por DWA siguiendo la ruta.

    dwa_traj.png – Visualización de la trayectoria del DWA sobre el mapa.

Personalización

    Cambia los puntos de inicio y meta ajustando x_start, y_start, x_goal, y_goal.

    Modifica el downsample_factor para controlar la resolución de la cuadrícula.

    Ajusta los parámetros de DWA en el diccionario params dentro del script para adaptarlos a tu robot.

    Puedes cambiar entre planificadores compatibles en SearchFactory (ej.: theta_star, a_star, dijkstra).

Estructura del proyecto

├── main.py
├── python_motion_planning/
│   ├── utils.py
│   ├── grid.py
│   └── search_factory.py
├── Oschersleben_map.yaml
├── Oschersleben_map.png
├── requirements.txt
└── results_theta_dwa/

Link de los videos 
    https://youtu.be/xQRZ2jU36v4
    https://youtu.be/4o3UPumjYtY
