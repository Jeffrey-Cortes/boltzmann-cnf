"""
main.py
=======
Punto de entrada. Solo CONFIG y orquestación.
Para cambiar de sistema: edita CONFIG["hamiltoniano"] y los parámetros relevantes.
Para agregar un sistema nuevo: toca fisica/ y analisis/plots.py, no este archivo.
"""

import os
import sys
import random
import numpy as np
from datetime import datetime

import time
import torch

from utils.logger         import Logger, guardar_config, crear_carpeta_corrida
from cnf.entrenamiento    import train
from cnf.muestreo         import sample, ground_truth_mcmc
from fisica.hamiltonianos import H_double_well, H_harmonic
from fisica.cristales     import make_H_debye_chain_1D, make_H_phi4_chain
from analisis.metricas    import (calcular_metricas, metrica_covarianza,
                                   valores_teoricos_double_well_nd,
                                   valores_teoricos_harmonic_nd,
                                   valores_teoricos_debye_chain_1D,
                                   observables_phi4)
from analisis.plots       import plot_training_curve, plot_comparacion, plot_por_sistema


# ═════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILIDAD
# ═════════════════════════════════════════════════════════════════════════════

def fijar_semilla(seed: int = 42):
    """
    Fija la semilla aleatoria global para reproducibilidad.
    Cubre Python, NumPy y PyTorch (CPU y GPU).
    La semilla queda registrada en config.txt para trazabilidad.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ═════════════════════════════════════════════════════════════════════════════
# TEMPORIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════

_t0 = {}

def tick(label: str):
    """Marca el inicio de una sección y la imprime."""
    _t0[label] = time.time()
    print(f"   {label}...")

def tock(label: str) -> float:
    """Marca el fin de una sección e imprime el tiempo transcurrido."""
    elapsed = time.time() - _t0.pop(label)
    print(f"   {label}: {elapsed:.1f}s")
    return elapsed


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    seed         = 42,
    hamiltoniano = "debye_chain_1D",  # double_well | harmonic | debye_chain_1D
                                # phi4_chain
    # Geometría
    dim          = 32,       # libre para harmonic/double_well; N para debye
    k            = 1,     # constante de resorte (debye_chain_1D)

    # Termodinámica
    beta         = 2.0,

    # Entrenamiento
    n_steps      = 10_000,
    batch_size   = 512,
    lr           = 3e-4,
    hidden       = 128,
    n_layers     = 3,
    n_hutchinson = 3,
    K            = 10,
    clip_grad    = 1.0,

    # Muestreo
    n_muestras   = 5000,
    mcmc_steps   = 100_000,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL HAMILTONIANO
# ═════════════════════════════════════════════════════════════════════════════

def construir_hamiltoniano(config):
    """
    Devuelve (H_func, Sigma_exact, plot_extras).
    Sigma_exact es None si el sistema no tiene solución analítica gaussiana.
    plot_extras es un dict con argumentos adicionales para plot_por_sistema.
    """
    ham  = config["hamiltoniano"]
    beta = config["beta"]
    dim  = config["dim"]

    if ham == "double_well":
        return H_double_well, None, {}

    elif ham == "harmonic":
        # Solución exacta: cada xᵢ ~ N(0, 1/β)  →  Σ = I/β
        Sigma_exact = np.eye(dim) / beta
        return H_harmonic, Sigma_exact, {}

    elif ham == "debye_chain_1D":
        H_func, Sigma_exact, meta = make_H_debye_chain_1D(
            dim, f=config["k"], beta=beta
        )
        return H_func, Sigma_exact, {"Sigma_exact": Sigma_exact, "meta": meta}
    elif ham == "phi4_chain":
        H_func, _, meta = make_H_phi4_chain(
            config["dim"], f=config["k"], beta=config["beta"]
        )
        return H_func, None, {"meta": meta}

    else:
        raise ValueError(f"Hamiltoniano desconocido: '{ham}'. "
                         f"Disponibles: double_well | harmonic | debye_chain_1D")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    fijar_semilla(CONFIG["seed"])
    _t_inicio = time.time()
    run_dir = crear_carpeta_corrida(BASE_DIR, ham=CONFIG["hamiltoniano"])
    logger  = Logger(os.path.join(run_dir, "run.log"))
    sys.stdout = logger

    try:
        print(f"Corrida : {run_dir}")
        print(f"Inicio  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        H_func, Sigma_exact, plot_extras = construir_hamiltoniano(CONFIG)
        guardar_config(run_dir, CONFIG)

        # ── Entrenamiento ─────────────────────────────────────────────────────
        print(f"\n=== Entrenando CNF — {CONFIG['hamiltoniano']}  β={CONFIG['beta']} ===")
        tick("Entrenamiento")
        net, losses = train(
            H_func       = H_func,
            dim          = CONFIG["dim"],
            beta         = CONFIG["beta"],
            n_steps      = CONFIG["n_steps"],
            batch_size   = CONFIG["batch_size"],
            lr           = CONFIG["lr"],
            hidden       = CONFIG["hidden"],
            n_layers     = CONFIG["n_layers"],
            n_hutchinson = CONFIG["n_hutchinson"],
            K            = CONFIG["K"],
            clip_grad    = CONFIG["clip_grad"],
        )
        model_path = os.path.join(run_dir, "modelo.pt")
        torch.save(net.state_dict(), model_path)
        print(f"Modelo guardado: {model_path}")
        tock("Entrenamiento")

        # ── Muestreo ──────────────────────────────────────────────────────────
        print("\n=== Generando muestras ===")
        tick("Muestreo red")
        samples_net = sample(
            net, n=CONFIG["n_muestras"],
            dim=CONFIG["dim"], K=CONFIG["K"]
        )
        tock("Muestreo red")
        tick("Muestreo MCMC")
        samples_mcmc = ground_truth_mcmc(
            H_func, CONFIG["beta"],
            n           = CONFIG["n_muestras"],
            dim         = CONFIG["dim"],
            steps       = CONFIG["mcmc_steps"],
            Sigma_exact = Sigma_exact,
        )

        tock("Muestreo MCMC")

        # ── Métricas de covarianza (si hay solución exacta) ───────────────────
        if Sigma_exact is not None:
            print("\n=== Métricas de covarianza ===")
            metrica_covarianza(samples_net,  Sigma_exact, run_dir, label="red")
            metrica_covarianza(samples_mcmc, Sigma_exact, run_dir, label="mcmc")

        # ── Figuras ───────────────────────────────────────────────────────────
        print("\n=== Generando figuras ===")
        plot_training_curve(losses, run_dir)
        plot_comparacion(samples_net, samples_mcmc, CONFIG["beta"], run_dir)
        plot_por_sistema(
            samples_net, samples_mcmc,
            CONFIG["beta"], CONFIG["hamiltoniano"],
            run_dir, **plot_extras
        )

        # ── Métricas cuantitativas ────────────────────────────────────────────
        print("\n=== Calculando métricas ===")
        calcular_metricas(
            samples_net, samples_mcmc,
            H_func, run_dir,
            beta=CONFIG["beta"]
        )

        # ── Valores teóricos específicos ──────────────────────────────────────
        if CONFIG["hamiltoniano"] == "harmonic":
            E_th, Cv_th = valores_teoricos_harmonic_nd(
                CONFIG["dim"], CONFIG["beta"]
            )
            print("\n=== Valores teóricos (harmonic) ===")
            print(f"  Energía media  ⟨H⟩ = {E_th:.4f}  [d/(2β)]")
            print(f"  Calor específico Cv = {Cv_th:.4f}  [d/2]")

        elif CONFIG["hamiltoniano"] == "double_well":
            E_th, Cv_th = valores_teoricos_double_well_nd(
                CONFIG["dim"], CONFIG["beta"]
            )
            print("\n=== Valores teóricos (double well) ===")
            print(f"  Energía media  ⟨H⟩ = {E_th:.4f}  [integración numérica]")
            print(f"  Calor específico Cv = {Cv_th:.4f}  [integración numérica]")

        elif CONFIG["hamiltoniano"] == "debye_chain_1D":
            E_th, Cv_th = valores_teoricos_debye_chain_1D(
                CONFIG["dim"], CONFIG["beta"]
            )
            print("\n=== Valores teóricos (debye_chain_1D) ===")
            print(f"  Energía media  ⟨H⟩ = {E_th:.4f}  [N/(2β)]")
            print(f"  Calor específico Cv = {Cv_th:.4f}  [N/2]")
        
        elif CONFIG["hamiltoniano"] == "phi4_chain":
            observables_phi4(
                samples_net, samples_mcmc,
                plot_extras["meta"], CONFIG["beta"], run_dir
            )

        print(f"\nFin     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tiempo total: {time.time() - _t_inicio:.1f}s")
        print(f"Archivos: {run_dir}")

    finally:
        sys.stdout = logger.terminal
        logger.close()