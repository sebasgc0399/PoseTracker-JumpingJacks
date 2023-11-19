import ultralytics
ultralytics.checks()

from camera_operations import CameraOperations
from pose_estimation import PoseEstimator
from visualization import Visualization
from jumping_jack_counter import JumpingJackCounter

jumpingJackCounter = JumpingJackCounter()

# Crear una instancia de CameraOperations
camera = CameraOperations()

# Crear una instancia de PoseEstimator
pose_estimator = PoseEstimator()

# Crear una instancia de Visualization
visualization = Visualization()

# Lista de conexiones entre puntos clave
# Cada conexión es un par de índices en la lista de puntos clave
conexiones = [
    (0, 1),  # Boca a Ojo Derecho
    (0, 2),  # Boca a Ojo Izquierdo
    (1, 4),  # Ojo Derecho a Oreja Derecha
    (2, 3),  # Ojo Izquierdo a Oreja Izquierda
    (4, 6),  # Oreja Derecha a Hombro Derecho
    (3, 5),  # Oreja Izquierda a Hombro Izquierdo
    (6, 8),  # Hombro Derecho a Codo Derecho
    (5, 7),  # Hombro Izquierdo a Codo Izquierdo
    (8, 10), # Codo Derecho a Muñeca Derecha
    (7, 9),  # Codo Izquierdo a Muñeca Izquierda
    (6, 12), # Hombro Derecho a Parte del Tronco Derecho
    (5, 6), # Hombro Izquierdo a Hombro Derecho
    (5, 11), # Hombro Izquierdo a Parte del Tronco Izquierdo
    (11, 12), # Parte del Tronco Izquierdo a Parte del Tronco Derecho
    (12, 14),# Parte del Tronco Derecho a Rodilla Derecha
    (11, 13),# Parte del Tronco Izquierdo a Rodilla Izquierda
    (14, 16),# Rodilla Derecha a Pie Derecho
    (13, 15),# Rodilla Izquierda a Pie Izquierdo
]

while True:
    # Capturar frame por frame
    frame, frame_rgb = camera.capture_frame()
    if frame is None:
        break

    # Realizar detecciones con YOLOv8
    resultados = pose_estimator.detectar_pose(frame_rgb)

    # Procesar los resultados
    if resultados[0].keypoints.xy[0].nelement() > 0:
        visualization.dibujar_keypoints(frame, resultados[0].keypoints.xy[0])
        visualization.dibujar_conexiones(frame, resultados[0].keypoints.xy[0], conexiones)
     
        # Determinar el estado actual y actualizar el contador
        estado_actual = pose_estimator.determinar_estado(resultados[0].keypoints.xy[0])
        print(f"Estado actual: {estado_actual}")  # Depuración


        contador_jumping_jacks = jumpingJackCounter.contar(estado_actual)

    # Obtener FPS y mostrarlos
    fps = camera.calculate_fps()
    visualization.mostrar_fps(frame, fps)

    # Mostrar el contador de Jumping Jacks
    visualization.mostrar_contador_jumpings(frame, contador_jumping_jacks)
    
    # Mostrar el frame resultante
    visualization.mostrar_frame(frame)

    # Romper el bucle si se presiona 'q'
    if not visualization.debe_continuar():
        break

# Cuando todo esté hecho, liberar la captura
camera.release()
visualization.cerrar()