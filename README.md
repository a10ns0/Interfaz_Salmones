# Interfaz de Visión Estereoscópica con ZED SDK y Python

Este repositorio contiene la implementación y documentación para el despliegue de interfaces de visión estereoscópica utilizando el **ZED SDK** y su wrapper de Python (`pyzed`). [cite_start]Esta interfaz actúa como un orquestador de recursos híbridos, gestionando la rectificación estereoscópica, el cálculo de disparidad y la fusión de sensores mediante aceleración por hardware (CUDA).

## 📋 Requisitos del Sistema

Debido a la profunda integración con el hardware, el entorno debe cumplir especificaciones estrictas antes de la instalación.

### Hardware
***GPU:** NVIDIA con Compute Capability > 5.0 (Series GTX 10, RTX 30/40, Quadro, Jetson).
***VRAM:** Mínimo 6 GB recomendado para modelos de profundidad NEURAL o resolución HD2K.
***CPU:** Procesador moderno x64 (Intel i5/i7 o AMD Ryzen 5/7).
***RAM:** 8 GB mínimo (16 GB recomendado para desarrollo).

### Software
***OS:** Windows 10/11 o Linux (Ubuntu 20.04/22.04).
***Arquitectura:** Estrictamente x64.
***Python:** Versión compatible con el SDK instalado (ver tabla abajo).

| Versión ZED SDK | Versiones Python Soportadas (x64) |
| :--- | :--- |
| **SDK 5.1/5.0** | 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14  |
| **SDK 4.x** | 3.7 - 3.11 |

---

## Guía de Instalación

La instalación sigue un orden jerárquico estricto de tres fases.

### Fase 1: Instalación del ZED SDK (Nivel Sistema)
1.  Descargue el instalador para su SO y versión de CUDA desde el sitio de Stereolabs.
2.  **Windows:** Ejecute el instalador y permita que descargue/instale el CUDA Toolkit si no está presente.
3.  **Linux:** Otorgue permisos (`chmod +x`) y ejecute el script `.run`.
4.  **Reinicio:** Es obligatorio reiniciar el equipo para cargar las variables de entorno.

### Fase 2: Entorno Virtual y Dependencias
Se recomienda usar un entorno virtual (Anaconda o venv) para aislar las librerías.

```bash
# Ejemplo con Conda
conda create --name zed_env python=3.9
conda activate zed_env
```

### Instale las dependencias críticas antes de compilar la API:

```bash
python -m pip install cython numpy opencv-python pyopengl
```
Nota: Cython >= 3.0.0 y NumPy >= 2.0 son requeridos para las versiones nuevas del SDK

### Fase 3: Instalación de la API

El paquete pyzed no está en pip; debe instalarse usando el script local get_python_api.py incluido en el SDK.

Ubicación del script:

1) Windows: C:\Program Files (x86)\ZED SDK\ 

2) Linux: /usr/local/zed/


**Instrucción Crítica para Windows:** No ejecute el script directamente en Program Files (causa error de permisos). Cópielo a sus Documentos primero:

```bash
# PowerShell
copy "C:\Program Files (x86)\ZED SDK\get_python_api.py" $HOME\Documents\
cd $HOME\Documents\
python get_python_api.py
```

## Solución de Problemas (Troubleshooting)
Error: ImportError: DLL load failed while importing sl
Este error es común en Windows e indica que Python no encuentra las DLLs de C++.

Soluciones:

Verifique que C:\Program Files (x86)\ZED SDK\bin esté en el PATH del sistema.


Hard Fix: Copie manualmente los archivos .dll desde la carpeta bin del SDK a la carpeta donde está su script .py.

Asegúrese de no mezclar versiones (ej. SDK compilado para CUDA 11 ejecutándose en drivers CUDA 12).


## Funcionalidad de la Interfaz
La interfaz permite reproducir e importar archivos .svo (Stereo Video Odometry). Esto permite desarrollar sin la cámara física conectada, simulando una entrada en vivo.
