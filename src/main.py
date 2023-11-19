import ultralytics
ultralytics.checks()

# Importar las bibliotecas necesarias
import cv2
from camera_operations import CameraOperations
from pose_estimation import PoseEstimator

# Crear una instancia de CameraOperations
camera = CameraOperations()

# Crear una instancia de PoseEstimator
pose_estimator = PoseEstimator()

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

contador_jumping_jacks = 0
estado_previo = "Cerrado"  


while True:
    # Capturar frame por frame
    frame, frame_rgb = camera.capture_frame()
    if frame is None:
        break

    # Realizar detecciones con YOLOv8
    resultados = pose_estimator.detectar_pose(frame_rgb)

    # Procesar los resultados
    if resultados[0].keypoints.xy[0].nelement() > 0:
        # Dibujar los keypoints
        for idx, punto in enumerate(resultados[0].keypoints.xy[0]):
            x, y = punto
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Dibujar las conexiones
        for conexion in conexiones:
            punto_inicio = resultados[0].keypoints.xy[0][conexion[0]]
            punto_final = resultados[0].keypoints.xy[0][conexion[1]]

            if punto_inicio[0] > 0 and punto_inicio[1] > 0 and punto_final[0] > 0 and punto_final[1] > 0:
                cv2.line(frame, (int(punto_inicio[0]), int(punto_inicio[1])), 
                         (int(punto_final[0]), int(punto_final[1])), (255, 0, 0), 2)
     
        # Determinar el estado actual y actualizar el contador
        estado_actual = pose_estimator.determinar_estado(resultados[0].keypoints.xy[0])
        print(f"Estado actual: {estado_actual}")  # Depuración

        # Cambiar la lógica para contar un Jumping Jack completo
        if estado_previo == "Abierto" and estado_actual == "Cerrado":
            contador_jumping_jacks += 1
            print(f"Jumping Jack contado! Total: {contador_jumping_jacks}")  # Depuración

        estado_previo = estado_actual

    # Mostrar el contador de Jumping Jacks
    cv2.putText(frame, f'Jumping Jacks: {contador_jumping_jacks}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar el frame resultante
    cv2.imshow('Detecciones de Postura', frame)

    # Romper el bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cuando todo esté hecho, liberar la captura
camera.release()
cv2.destroyAllWindows()