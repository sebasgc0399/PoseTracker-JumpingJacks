import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Analizar video o cámara en tiempo real.')
    parser.add_argument('--video', help='Nombre del archivo de video en Data/videos para analizar', default=None)
    return parser.parse_args()
