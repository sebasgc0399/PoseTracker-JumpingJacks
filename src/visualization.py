import cv2

class Visualization:
    @staticmethod
    def dibujar_keypoints(frame, keypoints):
        for idx, punto in enumerate(keypoints):
            x, y = punto
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    @staticmethod
    def dibujar_conexiones(frame, keypoints, conexiones):
        for conexion in conexiones:
            punto_inicio = keypoints[conexion[0]]
            punto_final = keypoints[conexion[1]]

            if punto_inicio[0] > 0 and punto_inicio[1] > 0 and punto_final[0] > 0 and punto_final[1] > 0:
                cv2.line(frame, (int(punto_inicio[0]), int(punto_inicio[1])), 
                         (int(punto_final[0]), int(punto_final[1])), (255, 0, 0), 2)

    @staticmethod
    def mostrar_contador_jumpings(frame, contador_jumping_jacks):
        cv2.putText(frame, f'Jumping Jacks: {contador_jumping_jacks}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    @staticmethod
    def mostrar_frame(frame, window_name='Detecciones de Postura'):
        cv2.imshow(window_name, frame)

    @staticmethod
    def debe_continuar():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True
    
    @staticmethod
    def mostrar_fps(frame, fps):
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    @staticmethod
    def cerrar():
        return cv2.destroyAllWindows()