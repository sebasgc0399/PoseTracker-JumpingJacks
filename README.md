# PoseTracker-JumpingJacks
## Descripción
**PoseTracker-JumpingJacks** es una aplicación innovadora que utiliza el poder del aprendizaje automático y la visión por computadora para rastrear y analizar movimientos humanos en tiempo real. Especialmente diseñada para monitorear y contar ejercicios de jumping jacks, esta herramienta es perfecta para entusiastas del fitness, entrenadores personales y desarrolladores de tecnología de seguimiento de movimiento.

## Requisitos del Sistema
- Python 3.8-3.11 (Nota: PyTorch no es compatible con versiones de Python posteriores a la 3.11 en Windows)

## Características
- **Detección de Posturas en Tiempo Real:** Utiliza modelos de YOLOv8 para detectar y rastrear posturas humanas con precisión.
- **Contador de Jumping Jacks:** Cuenta automáticamente los jumping jacks realizados por el usuario, facilitando el seguimiento del progreso del ejercicio.
- **Análisis de Movimiento:** Proporciona retroalimentación sobre la forma y la técnica del ejercicio.
- **Fácil de Usar:** Interfaz intuitiva y guía paso a paso para empezar rápidamente.

## Nueva Funcionalidad: Análisis de Videos
PoseTracker-JumpingJacks ahora permite cargar y analizar videos de ejercicios. Esta nueva característica es perfecta para revisar y mejorar la técnica de ejercicio o para fines de entrenamiento y análisis.

## Cómo Usar el Análisis de Videos
1. Coloca tus videos en la carpeta data/videos.

2. Ejecuta el programa con el comando de video:

```python main.py --video tu_video.mp4```

Reemplaza tu_video.mp4 con el nombre de tu archivo de video.

## Cómo Empezar
### Usando Conda
Si prefieres usar Conda, puedes crear un entorno y luego instalar las dependencias.

1. **Crear Entorno Conda:** ```conda env create -f environment.yml```
2. **Activar Entorno:** ```conda activate PoseTracker```
3. **Ejecutar la Aplicación:** ```python main.py```

### Usando Python Local
Para aquellos que usan Python directamente:

1. **Instalar Dependencias:** ```pip install -r requirements.txt```
2. **Ejecutar la Aplicación:** ```python main.py```

## Tecnologías Utilizadas
- Python
- OpenCV
- YOLO (You Only Look Once)
- Numpy

## Prueba
https://github.com/sebasgc0399/PoseTracker-JumpingJacks/assets/61243474/c595f2c8-1f35-41ff-a65e-584c28fcd168