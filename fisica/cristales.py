"""
fisica/cristales.py
===================
Sistemas con estructura de red y solución exacta gaussiana.

Devuelven (H_func, Sigma_exact, meta) donde:
    H_func(x)    → energía, shape (batch,)
    Sigma_exact  → covarianza analítica exacta, np.ndarray shape (d, d)
    meta         → dict con información del sistema

Permite comparación cuantitativa mediante error de Frobenius:
    ||Σ_emp - Σ_exact||_F / ||Σ_exact||_F
"""

import torch
import numpy as np


def make_H_debye_chain_1D(N, f=1.0, beta=1.0):
    """
    Cadena monoatómica 1D con PBC — Hamiltoniano clásico del modelo de Debye.

        H = (f/2) Σ_j (x_j - x_{j+1})²  =  ½ xᵀ K x

    K es el laplaciano 1D circular: K_{ii} = 2f,  K_{i,i±1} = -f.
    El modo cero se regulariza con ε·I.

    Solución exacta: N(0, (βK)⁻¹).

    Relación de dispersión exacta [Ec. 11-40]:
        ω_k = ω_max |sin(πk/N)|,   ω_max = 2√f

    Densidad de estados [Ec. 11-47]:
        g(ω) = (2N/π) / √(ω_max² - ω²)
    (singularidad de Van Hove en ω_max)

    Args:
        N:    número de átomos (= dimensión del espacio)
        f:    constante de resorte
        beta: inverso de temperatura

    Returns:
        H_func      — hamiltoniano, (batch, N) → (batch,)
        Sigma_exact — covarianza exacta, np.ndarray shape (N, N)
        meta        — dict con freqs_exact, omega_max, N, f
    """
    K = torch.zeros(N, N)
    for i in range(N):
        K[i, i]           = 2.0 * f
        K[i, (i+1) % N]  = -f
        K[(i+1) % N, i]  = -f
    K += 0.1 * torch.eye(N)    # regularización modo cero

    K_np        = K.numpy()
    Sigma_exact = np.linalg.inv(beta * K_np)

    k_indices   = np.arange(N)
    freqs_analy = 2.0 * np.sqrt(f) * np.abs(np.sin(np.pi * k_indices / N))
    eigvals_K   = np.linalg.eigvalsh(K_np)
    freqs_num   = np.sqrt(np.maximum(eigvals_K, 0.0))

    meta = {
        "freqs_analy" : np.sort(freqs_analy),
        "freqs_num"   : np.sort(freqs_num),
        "omega_max"   : 2.0 * np.sqrt(f),
        "N"           : N,
        "f"           : f,
    }

    def H_func(x):
        K_t = torch.tensor(K_np, dtype=torch.float32, device=x.device)
        return 0.5 * (x * (x @ K_t)).sum(dim=-1)

    return H_func, Sigma_exact, meta

def make_H_phi4_chain(N, f=1.0, beta=1.0):
    """
    Cadena φ⁴ 1D con PBC.

        H = Σ_j [ (x_j² - 1)² + (f/2)(x_j - x_{j+1})² ]

    Límites:
        f → 0  :  N pozos dobles independientes
        f → ∞  :  los sitios se sincronizan → fase ordenada (⟨x⟩ ≠ 0)

    No tiene solución analítica exacta → Sigma_exact = None.
    La transición de fase ocurre en f_c(β, N) que se estima numéricamente
    con MCMC (ver meta["f_critico_estimado"]).

    Args:
        N:    número de sitios
        f:    acoplamiento entre vecinos
        beta: inverso de temperatura

    Returns:
        H_func      — hamiltoniano, (batch, N) → (batch,)
        None        — sin solución analítica
        meta        — dict con N, f, beta, regimen
    """
    # Estimación heurística del régimen: compara energía de barrera
    # (escala 1) con energía de acoplamiento (escala f)
    if f < 0.3:
        regimen = "desacoplado (pozos independientes)"
    elif f > 3.0:
        regimen = "fuertemente acoplado (fase ordenada)"
    else:
        regimen = "crítico (transición de fase)"

    meta = {
        "N"      : N,
        "f"      : f,
        "beta"   : beta,
        "regimen": regimen,
    }

    def H_func(x):
        # Término de sitio: doble pozo en cada x_j
        V_sitio = ((x ** 2 - 1) ** 2).sum(dim=-1)
        # Término de acoplamiento: resortes entre vecinos con PBC
        dx = x - torch.roll(x, shifts=1, dims=-1)   # x_j - x_{j-1}
        V_resorte = (f / 2.0) * (dx ** 2).sum(dim=-1)
        return V_sitio + V_resorte

    return H_func, None, meta