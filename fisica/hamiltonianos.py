"""
fisica/hamiltonianos.py
=======================
Hamiltonianos de sistemas físicos simples.

Cada función recibe un tensor x de shape (batch, dim) y devuelve
un tensor de shape (batch,) con la energía de cada configuración.
"""

import torch


def H_double_well(x):
    """
    Doble pozo ND: H = Σᵢ (xᵢ²-1)²

    Cada dimensión tiene dos mínimos en xᵢ = ±1, separados por una barrera
    de altura 1. El sistema tiene 2^d mínimos degenerados en {±1}^d.

    Marginal exacta por dimensión (no gaussiana):
        p(xᵢ) ∝ exp(-β (xᵢ²-1)²)
    Las dimensiones son independientes bajo esta H, por lo que todas las
    marginales son idénticas.

    Solución exacta global: no disponible en forma cerrada (no gaussiano).
    Valores teóricos de ⟨H⟩ y Cᵥ: integración numérica 1D × dim
    (ver valores_teoricos_double_well_nd en analisis/metricas.py).
    """
    return ((x ** 2 - 1) ** 2).sum(dim=-1)


def H_harmonic(x):
    """
    Oscilador armónico ND: H = ‖x‖²/2

    Solución exacta: π_H = N(0, I/β).
    """
    return 0.5 * (x ** 2).sum(dim=-1)