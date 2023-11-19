import numpy as np
import math
import os
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self):
        self.historial_keypoints = []
        model_path = 'models/yolov8n-pose.pt'

        # Verificar si el modelo ya está descargado en la carpeta 'models'
        if os.path.exists(model_path):
            # Si está, cargar el modelo local
            self.model = YOLO(model_path)
        else:
            # Si no está, descargarlo (ultralytics lo descargará automáticamente)
            self.model = YOLO('yolov8n-pose.pt')
            # Considera mover el archivo descargado a la carpeta 'models' si es necesario

    def calcular_angulo(self, p1, p2, p3):
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
    
    def determinar_estado(self, keypoints, num_frames=5):
        # Asumiendo que los índices de los keypoints son los siguientes:
        # 6: Hombro Derecho, 5: Hombro Izquierdo
        # 10: Muñeca Derecha, 9: Muñeca Izquierda
        # 12: Tronco Derecho, 11: Tronco Izquierdo
        # 14: Rodilla Derecho, 13: Rodilla Izquierdo
        # 16: Pie Derecho, 15: Pie Izquierdo

        # Añadir los keypoints actuales al historial
        self.historial_keypoints.append(keypoints)
        if len(self.historial_keypoints) > num_frames:
            self.historial_keypoints.pop(0)

        # Calcular el promedio de los últimos 'num_frames' frames
        promedio_keypoints = np.mean(self.historial_keypoints, axis=0)

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
        angulo_derecha = self.calcular_angulo(tronco_derecha, rodilla_derecha, pie_derecho)
        print(f"Ángulo Pierna Derecha: {angulo_derecha} grados")

        # Calcular el ángulo de la pierna izquierda
        angulo_izquierdo = self.calcular_angulo(tronco_izquierda, rodilla_izquierda, pie_izquierdo)
        print(f"Ángulo Pierna Izquierda: {angulo_izquierdo} grados")

        if angulo_derecha is None or angulo_derecha is None:
            return "Indeterminado"
        
        piernas_abiertas = angulo_derecha > umbral_piernas and angulo_izquierdo > umbral_piernas

        # Determinar el estado
        if brazos_abiertos and piernas_abiertas:
            return "Abierto"
        else:
            return "Cerrado"

    def detectar_pose(self, frame_rgb):
        resultados = self.model(frame_rgb)
        return resultados

    def procesar_keypoints(self, keypoints):
        # Este método puede incluir cualquier procesamiento adicional que necesites hacer con los keypoints
        estado_actual = self.determinar_estado(keypoints, self.historial_keypoints)
        return estado_actual

# Otras funciones que necesites relacionadas con la estimación de pose
