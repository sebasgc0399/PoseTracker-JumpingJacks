class JumpingJackCounter:
    def __init__(self):
        self.contador = 0
        self.estado_previo = "Cerrado"

    def contar(self, estado_actual):
        if self.estado_previo == "Abierto" and estado_actual == "Cerrado":
            self.contador += 1
            print(f"Jumping Jack contado! Total: {self.contador}")  # Depuración
        self.estado_previo = estado_actual
        return self.contador
