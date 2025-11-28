Manual Técnico Integral: Arquitectura, Despliegue y Programación de Interfaces de Visión Estereoscópica con ZED SDK y Python


1. Fundamentos de la Arquitectura y Funcionalidad de la Interfaz

La implementación de sistemas de percepción espacial mediante visión estereoscópica pasiva representa uno de los desafíos más complejos en la ingeniería de visión por computador moderna. El SDK de ZED, desarrollado por Stereolabs, y su correspondiente interfaz de programación en Python (pyzed), constituyen una solución de middleware sofisticada que abstrae la complejidad matemática de la triangulación de píxeles y la aceleración por hardware, permitiendo a los desarrolladores integrar capacidades de detección de profundidad, seguimiento posicional y mapeo espacial en entornos de alto nivel.
Este manual técnico tiene como propósito desglosar de manera exhaustiva los mecanismos operativos de la interfaz, los protocolos de instalación y configuración del entorno, la gestión de dependencias y los flujos de ejecución crítica. A diferencia de una simple librería de procesamiento de imágenes, la interfaz ZED opera como un orquestador de recursos híbridos, gestionando tensores de memoria tanto en la CPU (Host) como en la GPU (Device), lo que implica una comprensión profunda de su arquitectura subyacente para una implementación exitosa.

1.1 Naturaleza y Propósito de la Interfaz pyzed

La pregunta fundamental sobre "qué es lo que hace la interfaz" requiere una disección técnica de su rol dentro de la pila de software de robótica y visión. La interfaz Python para ZED, distribuida bajo el paquete pyzed, no es una reimplementación del núcleo de procesamiento en Python puro; es un wrapper (envoltorio) de alto rendimiento compilado mediante Cython.
El núcleo del ZED SDK está escrito en C++ altamente optimizado, diseñado para interactuar directamente con la API de CUDA de NVIDIA para realizar operaciones de álgebra lineal masivamente paralelas. La interfaz pyzed actúa como un puente de enlace (binding) que expone estas funciones C++ al intérprete de Python.1
Funciones Críticas de la Interfaz:
Rectificación y Sincronización Estereoscópica: La interfaz toma las señales de video crudas de los sensores izquierdo y derecho de la cámara (que sufren de distorsión de lente radial y tangencial) y aplica una rectificación matemática en tiempo real. Esto alinea las líneas epipolares de ambas imágenes, un prerrequisito matemático indispensable para calcular la disparidad y, consecuentemente, la profundidad.
Cálculo de Disparidad Acelerado por GPU: La función primordial de la interfaz es convertir la disparidad (la diferencia en la posición horizontal de un píxel entre la imagen izquierda y derecha) en una métrica de profundidad ($Z$). La fórmula subyacente que la interfaz resuelve millones de veces por segundo es $Z = \frac{f \cdot B}{d}$, donde $f$ es la distancia focal, $B$ es la línea base (distancia entre sensores) y $d$ es la disparidad. La interfaz delega este cálculo masivo a los núcleos CUDA de la GPU, liberando al procesador central para la lógica de la aplicación.
Fusión de Sensores (Sensor Fusion): En modelos como la ZED 2, ZED 2i o ZED X, la interfaz no solo captura luz. Gestiona un flujo de datos concurrente proveniente de una Unidad de Medición Inercial (IMU) interna, que incluye acelerómetros, giroscopios, barómetros y magnetómetros. La interfaz pyzed aplica filtros de Kalman extendidos o complementarios para fusionar estos datos inerciales con la odometría visual, proporcionando una estimación de pose (posición y orientación) robusta y de baja deriva.2
Gestión de Memoria Híbrida (sl.Mat): La interfaz introduce una estructura de datos propietaria denominada sl.Mat. A diferencia de los arrays de NumPy que residen en la memoria RAM del sistema, un sl.Mat puede asignar memoria en la VRAM de la GPU o en la RAM de la CPU. La interfaz gestiona las transferencias de memoria (marshalling) entre estos dos espacios, permitiendo que los datos de profundidad permanezcan en la GPU para su procesamiento con redes neuronales sin incurrir en la latencia del bus PCIe, a menos que el usuario solicite explícitamente una descarga a la CPU.3

1.2 El Paradigma de Ejecución SVO

Una capacidad distintiva que la interfaz habilita es el manejo del formato SVO (Stereo Video Odometry). La interfaz permite abstraer la fuente de entrada: para el código de aplicación, es transparente si los datos provienen de un bus USB 3.0 conectado a una cámara física en tiempo real o de un archivo .svo almacenado en disco.
Cuando la interfaz "ejecuta" un archivo SVO, simula una cámara virtual. Inyecta los pares de imágenes estéreo y los metadatos de los sensores con la misma cadencia temporal y estructura de datos que tendría una transmisión en vivo. Esto permite que la interfaz sea utilizada no solo para operación, sino para simulación, validación y desarrollo offline, garantizando que el comportamiento del algoritmo de visión sea determinista y reproducible.5

2. Matriz de Compatibilidad y Requisitos del Sistema

La instalación del ecosistema ZED es estrictamente jerárquica y sensible a versiones. Debido a la profunda integración con el hardware de NVIDIA, no existe flexibilidad en cuanto a las arquitecturas soportadas: el entorno debe cumplir con especificaciones rígidas antes de intentar cualquier importación de librerías.

2.1 Requisitos de Hardware y Arquitectura de Procesamiento

El prerrequisito no negociable para el funcionamiento de la interfaz es la presencia de una tarjeta gráfica NVIDIA compatible con CUDA. La interfaz depende de los binarios de CUDA para ejecutar los kernels de estereoscopía y redes neuronales (para los modos de profundidad NEURAL o ULTRA).
Arquitectura de GPU: Se requiere una GPU NVIDIA con una capacidad de cómputo (Compute Capability) superior a 5.0. Esto abarca desde la serie GeForce GTX 10 (Pascal) hasta las series RTX 30/40 (Ampere/Ada Lovelace) y las arquitecturas profesionales como Quadro o las integradas en la familia Jetson.7
Memoria de Video (VRAM): La cantidad de VRAM dicta la resolución máxima y el modelo de profundidad que la interfaz puede ejecutar. Para resoluciones HD2K o modelos de profundidad NEURAL, se recomiendan al menos 6 GB de VRAM.
CPU y RAM: Aunque la GPU realiza el trabajo pesado de visión, la interfaz requiere un procesador moderno (Intel i5/i7 o AMD Ryzen 5/7 de generaciones recientes) para gestionar el flujo de datos y la descodificación USB. La memoria RAM del sistema debe ser suficiente para almacenar los buffers de imágenes; 8 GB es el mínimo operativo, pero 16 GB es el estándar recomendado para desarrollo.7

2.2 Selección de la Versión de Python

La elección de la versión de Python no es arbitraria; está vinculada a la versión del ZED SDK que se pretende instalar y, en el caso de sistemas embebidos, a la versión del sistema operativo (JetPack).
Tabla de Compatibilidad de Python según Versión del SDK:

Versión del ZED SDK
Versiones de Python Soportadas (x64)
Notas de Compatibilidad
SDK 5.1 / 5.0
Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14
Versiones actuales y estables.1 Soporta las últimas características de NumPy 2.0.
SDK 4.x
Python 3.7 - 3.11
Versiones legacy comunes en sistemas industriales estabilizados.8
SDK 3.x
Python 3.5 - 3.9
Obsoleto para nuevos desarrollos, pero presente en hardware antiguo.

Restricciones Críticas:
Arquitectura x64: Es imperativo utilizar la versión de 64 bits de Python. Las distribuciones de 32 bits (x86) son incompatibles con los binarios de CUDA y el SDK, y su uso resultará en errores inmediatos de ImportError: DLL load failed o incompatibilidad de arquitectura.10
Coherencia con el Sistema Operativo:
Windows 10/11: El usuario tiene libertad para instalar cualquier versión de Python dentro del rango soportado (3.8-3.14). Se recomienda encarecidamente el uso de Python 3.10 o 3.11 para equilibrar estabilidad y soporte de librerías modernas.11
Linux (Ubuntu 20.04/22.04): Generalmente se prefiere la versión de Python nativa del sistema operativo para evitar conflictos con librerías del sistema (/usr/lib), aunque el uso de entornos virtuales (venv o conda) permite instalar versiones más recientes.
NVIDIA Jetson: En estas plataformas, la versión de Python está frecuentemente atada a la versión de JetPack instalada. Por ejemplo, JetPack 5.x suele estandarizar en Python 3.8, mientras que JetPack 6.x se mueve a versiones más nuevas. Intentar forzar una versión diferente en Jetson puede romper los enlaces con las librerías de aceleración de hardware de Tegra.12

3. Protocolos de Instalación y Despliegue del Entorno

La instalación de la interfaz se compone de dos fases secuenciales: la instalación del controlador y SDK a nivel de sistema, y la instalación del API de Python en el entorno de usuario. Invertir este orden o omitir pasos resultará en un entorno inoperante.

3.1 Fase I: Instalación del ZED SDK (Nivel Sistema)

Antes de escribir una sola línea de código en Python, el sistema operativo debe ser capaz de reconocer la cámara y ejecutar código CUDA.

Procedimiento para Entornos Windows

Descarga del Instalador: Acceda al centro de desarrolladores de Stereolabs y descargue el instalador ejecutable (.exe) correspondiente a su versión de Windows y la versión de CUDA que desea utilizar (por ejemplo, ZED SDK for Windows 10/11 - CUDA 12.1).
Ejecución con Privilegios: Ejecute el instalador. El asistente realizará comprobaciones de hardware. Si no detecta una instalación de CUDA Toolkit compatible, ofrecerá descargarla e instalarla automáticamente. Acepte esta opción si no es un usuario avanzado que gestiona múltiples versiones de CUDA manualmente.7
Configuración de Variables de Entorno: El instalador añadirá automáticamente las rutas críticas al PATH del sistema (e.g., C:\Program Files (x86)\ZED SDK\bin). Estas rutas son esenciales para que Python encuentre las DLLs (librerías de enlace dinámico) en tiempo de ejecución.
Reinicio Obligatorio: Tras finalizar, es mandatorio reiniciar el equipo. Sin este paso, las variables de entorno no se recargan y la detección de la cámara por USB puede fallar.7

Procedimiento para Entornos Linux (Ubuntu/Debian)

Obtención del Binario: Descargue el archivo .run específico para su distribución (e.g., Ubuntu 22.04).
Permisos y Ejecución: Desde la terminal, navegue a la carpeta de descarga y otorgue permisos de ejecución:
Bash
chmod +x ZED_SDK_Ubuntu22_cuda12.1_v4.x.x.run


./ZED_SDK_Ubuntu22_cuda12.1_v4.x.x.run
```
3. Interacción: El script interactivo le preguntará si desea instalar dependencias, configurar reglas udev (para permisos de USB sin root) e instalar la API de Python. Aunque el script puede instalar la API de Python en el sistema global, se recomienda declinar esa opción específica para instalarla manualmente en un entorno virtual controlado posteriormente.14

3.2 Fase II: Preparación del Entorno de Desarrollo Python

El uso de entornos virtuales es una práctica estándar en la industria para aislar las dependencias del proyecto.
Opción A: Anaconda (Recomendado para Ciencia de Datos)
Anaconda gestiona paquetes binarios complejos, lo que facilita la instalación de dependencias científicas.

Bash


conda create --name zed_env python=3.10
conda activate zed_env


.15
Opción B: Venv (Nativo de Python)
Más ligero y directo para implementaciones puras de Python.

Bash


python -m venv zed_env
# Windows:
.\zed_env\Scripts\activate
# Linux:
source zed_env/bin/activate



3.3 Fase III: Instalación de la API pyzed

Este es el paso crítico donde se vincula el entorno Python con el SDK instalado en la Fase I. Stereolabs no publica el paquete pyzed en el índice público de PyPI (pip) debido a sus dependencias estrictas con el hardware local. En su lugar, proporciona un script de instalación local llamado get_python_api.py.

1. Localización del Script de Instalación

El script get_python_api.py se instala físicamente en el disco duro durante la Fase I.
Ruta en Windows: C:\Program Files (x86)\ZED SDK\.11
Ruta en Linux: /usr/local/zed/.1

2. Instalación de Dependencias Previas

Antes de ejecutar el script del wrapper, el entorno de Python debe tener instaladas las librerías fundamentales para compilar extensiones C y manejar matrices numéricas.
Ejecute el siguiente comando en su entorno activado:

Bash


python -m pip install cython numpy opencv-python pyopengl


Cython: Requerido (versión >= 3.0.0) para compilar o enlazar los módulos C++.1
NumPy: Requerido (versión >= 2.0 recomendada para SDKs nuevos) para la estructura de datos de las imágenes.1

3. Ejecución del Script get_python_api.py

El método de ejecución varía ligeramente según el sistema operativo debido a los permisos de escritura en directorios del sistema.
En Windows:
El directorio Program Files (x86) está protegido contra escritura para usuarios estándar. Ejecutar el script directamente allí suele causar errores de Permission denied cuando el script intenta descargar el archivo .whl temporalmente.
Método Correcto: Copie el archivo a un directorio de usuario y ejecútelo desde allí.
Abra una terminal (CMD o PowerShell) con su entorno virtual activado.
Copie el archivo:
PowerShell
copy "C:\Program Files (x86)\ZED SDK\get_python_api.py" %USERPROFILE%\Documents\


Navegue y ejecute:
PowerShell
cd %USERPROFILE%\Documents\
python get_python_api.py


.10
En Linux:
Puede ejecutarse directamente desde el directorio de instalación, asumiendo que el usuario tiene permisos de lectura/ejecución o usando sudo si se instala en el sistema global (aunque para entornos virtuales, no se debe usar sudo).

Bash


cd /usr/local/zed/
python get_python_api.py


Qué hace este script internamente:
Detecta la arquitectura del sistema (x64, aarch64).
Detecta la versión de Python activa (e.g., 3.10).
Detecta la versión instalada del ZED SDK y CUDA (e.g., 4.1, CUDA 12).
Construye una URL para descargar el archivo wheel (.whl) precompilado específico para esa combinación exacta desde los servidores de Stereolabs.
Instala el paquete descargado usando pip.
Si no encuentra un binario compatible, intenta compilarlo desde el código fuente (lo cual requiere un compilador C++ como Visual Studio Build Tools en Windows o GCC en Linux).11

4. Gestión de Librerías: Importación y Dependencias

Una vez instalado el entorno, la estructura del código Python debe seguir pautas específicas de importación para garantizar el acceso a las funciones de la cámara y la interoperabilidad con otras herramientas de visión.

4.1 Librerías Esenciales a Importar

El encabezado de su script de instalación o ejecución debe incluir las siguientes directivas de importación, cada una con un propósito funcional distinto:

Python


import sys
import pyzed.sl as sl
import numpy as np
import cv2
import threading  # Opcional, para captura asíncrona avanzada



Análisis Detallado de las Importaciones:

import pyzed.sl as sl
Identidad: Esta es la puerta de entrada al ZED SDK. El módulo pyzed contiene el sub-módulo sl (siglas de Stereolabs), que encapsula todas las clases y métodos.
Propósito: Proporciona acceso a:
sl.Camera: La clase controladora principal del dispositivo.
sl.InitParameters: Configuración inicial (resolución, FPS, sistema de coordenadas).
sl.RuntimeParameters: Configuración dinámica cuadro a cuadro (umbrales de confianza).
sl.Mat: El contenedor de datos de imagen híbrido CPU/GPU.
sl.Pose, sl.SensorsData: Estructuras para datos posicionales e inerciales.
Por qué es crucial: Sin esta librería, no existe comunicación con el controlador de la cámara.1
import numpy as np
Identidad: La librería estándar para computación numérica en Python.
Propósito: Sirve como el "lenguaje común" para el intercambio de datos. Aunque ZED usa sl.Mat internamente para eficiencia en GPU, cualquier operación de visión externa (como mostrar la imagen en pantalla con OpenCV, guardar un archivo con PIL, o procesar tensores con PyTorch) requiere que los datos se conviertan a un numpy.array.
Mecanismo: El método sl.Mat.get_data() extrae los píxeles brutos y los empaqueta en un array de NumPy.3
import cv2 (OpenCV)
Identidad: Librería de Visión por Computador de Código Abierto.
Propósito: La interfaz ZED no incluye herramientas de visualización de ventanas (GUI). Para ver lo que la cámara está capturando en tiempo real, dibujar cajas delimitadoras, o gestionar la entrada del teclado (para salir del programa), se utiliza OpenCV.
Integración: OpenCV recibe los arrays de NumPy generados por el SDK y los renderiza en ventanas del sistema operativo.3

4.2 Solución de Conflictos de Importación (Troubleshooting)

Uno de los errores más documentados en la fase de importación es ImportError: DLL load failed while importing sl. Esto es particularmente prevalente en Windows.
Diagnóstico y Solución:
Este error indica que aunque Python encontró el archivo sl.pyd (la librería Python), el sistema operativo no pudo encontrar las dependencias C++ que este requiere (sl_zed64.dll, nvcuda.dll, etc.).
Verificar PATH: Asegúrese de que C:\Program Files (x86)\ZED SDK\bin está en las variables de entorno del sistema.
Solución "Hard Fix": Copie manualmente todas las .dll desde la carpeta bin del SDK a la carpeta donde reside su script .py o a la carpeta site-packages/pyzed dentro de su entorno Python. Esto fuerza a Python a encontrar las librerías en el directorio local.19
Reinstalación de Entorno: Si el error persiste, suele deberse a una mezcla de versiones (e.g., instaló SDK para CUDA 11 pero tiene drivers de CUDA 12). Reinstale el SDK asegurando la coincidencia exacta de versiones.

5. Guía de Ejecución y Programación del Flujo de Trabajo

Para "ejecutar" la interfaz, no basta con llamar a una función. Se debe implementar un ciclo de vida de software que gestione la inicialización del hardware, la captura cíclica y la liberación de memoria. A continuación se detalla el código estructural necesario para un script de ejecución completo.

5.1 Fase de Inicialización

Antes de abrir la cámara, se debe configurar el objeto InitParameters. Estos parámetros son estáticos; una vez que la cámara se abre con ellos, no pueden cambiarse sin cerrar y reabrir la conexión.

Python


# Instanciación de parámetros
init = sl.InitParameters()

# 1. Configuración de Resolución y Tasa de Refresco
# Opciones: HD2K, HD1080, HD720, VGA. 
# La elección afecta el Campo de Visión (FOV) y la carga de CPU/GPU.
init.camera_resolution = sl.RESOLUTION.HD720
init.camera_fps = 60  # Debe ser compatible con la resolución (e.g., 2K está limitado a 15fps)

# 2. Configuración del Modo de Profundidad
# PERFORMANCE: Rápido, menor precisión en bordes.
# ULTRA: Alta precisión, mayor coste computacional.
# NEURAL: Máxima calidad usando IA, requiere GPU potente.
init.depth_mode = sl.DEPTH_MODE.NEURAL

# 3. Unidades y Coordenadas
init.coordinate_units = sl.UNIT.METER  # Resultados en metros
init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # Estándar para OpenGL/Robótica



5.2 Apertura del Dispositivo

La llamada zed.open(init) es bloqueante y realiza la conexión USB, carga los archivos de calibración desde el firmware de la cámara y reserva la memoria en la GPU.

Python


zed = sl.Camera()
status = zed.open(init)
if status!= sl.ERROR_CODE.SUCCESS:
    print(f"Fallo al abrir la cámara: {repr(status)}")
    exit(1)


Si se detecta un error, el código de estado (status) indicará la causa: CAMERA_NOT_DETECTED (USB desconectado), SENSOR_NOT_DETECTED (IMU fallida), o INVALID_CALIBRATION_FILE (descarga de internet necesaria).21

5.3 Bucle de Captura y Recuperación de Datos

El núcleo del programa es un bucle while infinito que solicita fotogramas constantemente. El proceso se divide en dos pasos: grab() (procesamiento) y retrieve() (extracción).

El Método grab()


Python


runtime_params = sl.RuntimeParameters()
err = zed.grab(runtime_params)


La función grab() es el disparador del motor de estereoscopía. Toma las imágenes crudas más recientes, las rectifica y calcula el mapa de disparidad en la GPU. Esta función sincroniza el hilo de Python con la tasa de fotogramas de la cámara.

Recuperación de Imágenes (retrieve_image)

Para obtener datos visuales (RGB, RGBA), se utiliza retrieve_image.

Python


image_sl = sl.Mat()  # Contenedor ZED
zed.retrieve_image(image_sl, sl.VIEW.LEFT) # Extraer vista izquierda
image_np = image_sl.get_data() # Conversión a NumPy (Copia de memoria si va a CPU)



Recuperación de Métricas (retrieve_measure)

Para obtener datos geométricos (Profundidad, Nube de Puntos), se utiliza retrieve_measure.

Python


depth_sl = sl.Mat()
zed.retrieve_measure(depth_sl, sl.MEASURE.DEPTH) # Mapa de profundidad (float32)
depth_np = depth_sl.get_data()


Insight Técnico: El mapa de profundidad contiene valores NAN (Not a Number) o inf (Infinito) donde la cámara no pudo calcular la distancia (oclusiones, falta de textura). Es crítico manejar estos valores antes de operar matemáticamente con ellos, reemplazándolos con ceros o interpolándolos.18

5.4 Visualización y Control

Usando OpenCV para mostrar los resultados:

Python


cv2.imshow("Vista ZED", image_np)
key = cv2.waitKey(1)
if key == 113: # Tecla 'q' para salir
    break



5.5 Cierre y Liberación

Al salir del bucle, es vital cerrar la cámara para apagar los sensores y liberar el contexto de CUDA.

Python


zed.close()



6. Operaciones Avanzadas: Archivos SVO y Grabación

La interfaz permite desacoplar la ejecución del hardware físico mediante archivos SVO. Esto es fundamental para "ejecutar" el software en máquinas sin cámara conectada o para reproducir escenarios específicos repetidamente.

6.1 Cómo Ejecutar un Archivo SVO

Para que la interfaz lea de un archivo en lugar de la cámara USB, se modifica el parámetro init_params antes de abrir la cámara.

Python


input_path = "ruta/a/mi_grabacion.svo"
init.set_from_svo_file(input_path)
init.svo_real_time_mode = False # False = Procesar a máxima velocidad posible


El resto del código (grab, retrieve) permanece idéntico. La interfaz inyectará los cuadros del archivo como si vinieran del sensor. El modo svo_real_time_mode = True respetará los tiempos originales de grabación (simulando en vivo), mientras que False procesará tan rápido como la GPU permita (ideal para procesar datasets masivos rápidamente).5

6.2 Grabación de SVO

Para generar estos archivos, se habilita el módulo de grabación:

Python


recording_param = sl.RecordingParameters("salida.svo", sl.SVO_COMPRESSION_MODE.H264)
zed.enable_recording(recording_param)


Durante el bucle grab(), cada cuadro procesado se escribe automáticamente en el archivo. Se recomienda usar compresión H.265 (HEVC) si la GPU lo soporta, ya que ofrece la mejor relación calidad/tamaño manteniendo una carga baja en la CPU gracias a la codificación por hardware NVENC.6

7. Diagnóstico, Rendimiento y Mejores Prácticas


7.1 Latencia en la Transferencia de Memoria

Un cuello de botella común es la latencia al llamar a get_data(). Transferir una imagen 2K de la VRAM (GPU) a la RAM (CPU) a través del bus PCIe es costoso.
Optimización: Si su procesamiento posterior (e.g., inferencia de red neuronal) puede hacerse en GPU, utilice librerías como CuPy. pyzed soporta la transferencia directa de punteros de memoria a CuPy, evitando la copia a la CPU por completo.1

7.2 Concurrencia y Multithreading

Python posee un Bloqueo Global de Intérprete (GIL). Si intenta ejecutar el procesamiento de la ZED en un hilo y otra tarea pesada en otro hilo usando la librería threading estándar, es posible que no vea mejoras de rendimiento y la tasa de fotogramas caiga.
Estrategia: La función grab() ya libera el GIL durante su ejecución en C++. Sin embargo, para paralelismo real en tareas de CPU, prefiera multiprocessing. Para tareas de inferencia de IA, asegúrese de que el motor de inferencia (PyTorch/TensorRT) esté configurado para ejecutarse asíncronamente en la GPU.23

7.3 Problemas con IMU en SVO

Al reproducir archivos SVO antiguos o grabados en versiones diferentes del SDK, los datos de la IMU (acelerómetro/giroscopio) pueden presentar jitter o desincronización. Asegúrese de actualizar el SDK a la versión 4.1 o superior, que introdujo el formato SVO v2 con mejoras significativas en la serialización de sensores de alta frecuencia.2
Resumen Ejecutivo de Implementación:
Validar Hardware: GPU NVIDIA (Compute > 5.0) y CPU x64.
Instalar Drivers: ZED SDK + CUDA (Reiniciar PC).
Configurar Python: Instalar numpy, cython, opencv-python.
Compilar Interfaz: Ejecutar get_python_api.py (ubicado en la carpeta de instalación del SDK).
Codificar: Importar pyzed.sl, inicializar sl.Camera, y ejecutar bucle grab/retrieve.
Siguiendo esta arquitectura, la interfaz proporcionará un flujo estable de datos espaciales listos para aplicaciones de robótica avanzada, realidad mixta o análisis volumétrico.
