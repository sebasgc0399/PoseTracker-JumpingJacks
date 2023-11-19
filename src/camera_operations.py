import cv2
import time

class CameraOperations:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.time_previous = 0
        self.fps = 0

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Calcular FPS
        time_current = time.time()
        self.fps = 1 / (time_current - self.time_previous)
        self.time_previous = time_current

        # Convertir el frame a formato RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, frame_rgb

    def calculate_fps(self):
        return self.fps

    def release(self):
        self.cap.release()
