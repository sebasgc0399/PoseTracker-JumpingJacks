import ultralytics
ultralytics.checks()

# Importar las bibliotecas necesarias
import cv2
import time
import numpy as np
from ultralytics import YOLO
import math

def calcular_angulo(p1, p2, p3):
    # Convertir a arrays de NumPy
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    # Comprobar si alguno de los puntos es (0, 0)
    if np.any(p1 == 0) or np.any(p2 == 0) or np.any(p3 == 0):
        # Devuelve None o maneja el caso como prefieras
        return None
    
    # Extraer coordenadas
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calcular los lados del triángulo
    a = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    b = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    c = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)

    # Calcular el ángulo en radianes
    angulo = math.acos((a**2 + b**2 - c**2) / (2 * a * b))

    # Convertir a grados
    angulo = math.degrees(angulo)

    return angulo


# Variables para suavizar los keypoints (por ejemplo, promedio móvil)
historial_keypoints = []

def determinar_estado(keypoints, historial, num_frames=5):
    # Asumiendo que los índices de los keypoints son los siguientes:
    # 6: Hombro Derecho, 5: Hombro Izquierdo
    # 10: Muñeca Derecha, 9: Muñeca Izquierda
    # 12: Tronco Derecho, 11: Tronco Izquierdo
    # 14: Rodilla Derecho, 13: Rodilla Izquierdo
    # 16: Pie Derecho, 15: Pie Izquierdo

    # Añadir los keypoints actuales al historial
    historial.append(keypoints)
    if len(historial) > num_frames:
        historial.pop(0)

    # Calcular el promedio de los últimos 'num_frames' frames
    promedio_keypoints = np.mean(historial, axis=0)

    # Obtener las coordenadas de los keypoints relevantes
    hombro_derecho, hombro_izquierdo = promedio_keypoints[6], promedio_keypoints[5]
    muneca_derecha, muneca_izquierda = promedio_keypoints[10], promedio_keypoints[9]
    tronco_derecha, tronco_izquierda = promedio_keypoints[12], promedio_keypoints[11]
    rodilla_derecha, rodilla_izquierda = promedio_keypoints[14], promedio_keypoints[13]
    pie_derecho, pie_izquierdo = promedio_keypoints[16], promedio_keypoints[15]

    # Agregar impresiones de depuración para ver las coordenadas
    print(f"Hombro Derecho: {hombro_derecho}, Muñeca Derecha: {muneca_derecha}")
    print(f"Hombro Izquierdo: {hombro_izquierdo}, Muñeca Izquierda: {muneca_izquierda}")
    print(f"tronco Derecha: {tronco_derecha}, Rodilla Derecho: {rodilla_derecha}, Pie Derecho: {pie_derecho}")
    print(f"tronco Izquierda: {tronco_izquierda}, Rodilla Izquierda: {rodilla_izquierda}, Pie Izquierdo: {pie_izquierdo}")

    # Ajustar estos umbrales según sea necesario
    umbral_brazos = 20  # Ajustar este valor
    umbral_piernas = 170  # Ajustar este valor

    # Condición modificada para brazos abiertos
    brazos_abiertos = muneca_derecha[1]+umbral_brazos < hombro_derecho[1] and muneca_izquierda[1]+umbral_brazos < hombro_izquierdo[1]

    # Condición modificada para piernas abiertas
    #piernas_abiertas = pie_derecho[0]+umbral_piernas > tronco_derecha[0] and pie_izquierdo[0]+umbral_piernas < tronco_izquierda[0]
    # Calcular el ángulo de la pierna derecha
    angulo_derecha = calcular_angulo(tronco_derecha, rodilla_derecha, pie_derecho)
    print(f"Ángulo Pierna Derecha: {angulo_derecha} grados")

    # Calcular el ángulo de la pierna izquierda
    angulo_izquierdo = calcular_angulo(tronco_izquierda, rodilla_izquierda, pie_izquierdo)
    print(f"Ángulo Pierna Izquierda: {angulo_izquierdo} grados")

    if angulo_derecha is None or angulo_derecha is None:
        return "Indeterminado"
    
    piernas_abiertas = angulo_derecha > umbral_piernas and angulo_izquierdo > umbral_piernas

    # Determinar el estado
    if piernas_abiertas:
        return "Abierto"
    else:
        return "Cerrado"

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # '0' suele ser la cámara web integrada


# Configurar y cargar el modelo YOLOv8
model = YOLO('yolov8n-pose.pt')  # build a new model from YAML

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

# Para calcular FPS
tiempo_previo = 0

contador_jumping_jacks = 0
estado_previo = "Cerrado"  


while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Marcar el tiempo actual
    tiempo_actual = time.time()

    # Calcular los FPS
    fps = 1 / (tiempo_actual - tiempo_previo)
    tiempo_previo = tiempo_actual

    # Convertir el frame a formato RGB para procesarlo con el modelo
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar detecciones con YOLOv8
    resultados = model(frame_rgb)

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
        estado_actual = determinar_estado(resultados[0].keypoints.xy[0], historial_keypoints)
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
cap.release()
cv2.destroyAllWindows()