"""
utils/logger.py
===============
Logger, guardar_config y crear_carpeta_corrida.
"""

import os
import sys
from datetime import datetime


class Logger:
    """Redirige stdout simultáneamente a consola y a archivo."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def guardar_config(run_dir, config):
    path = os.path.join(run_dir, "config.txt")
    with open(path, "w") as f:
        f.write("=== Configuración de la corrida ===\n\n")
        for k, v in config.items():
            f.write(f"{k:20s}: {v}\n")
    print(f"Guardado: {path}")


def crear_carpeta_corrida(base_dir, ham):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, "resultados", f"{ham}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir