"""
cnf/modelo.py
=============
Arquitectura de la red y estimador de Hutchinson.
"""

import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    """
    Red que aprende el campo de velocidades v_θ(x).

    Sin dependencia en t: el campo es estático (misma función en todos
    los pasos de Euler). Esto refleja la inspiración de Rectified Flow
    de buscar transportes simples y casi lineales.

    Arquitectura: MLP con activaciones SiLU.
    Input/Output: R^d → R^d

    Args:
        dim:      dimensión del espacio de configuraciones
        hidden:   neuronas por capa oculta
        n_layers: número de capas ocultas
    """

    def __init__(self, dim, hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def hutchinson_trace(v, x, n_samples=3):
    """
    Estima tr(∂v/∂x) via el estimador de Hutchinson:
        tr(A) = E_ε[εᵀ A ε],  ε ~ N(0, I)

    Costo O(d) en vez de O(d²) que costaría calcular el jacobiano completo.
    n_samples controla la varianza: más muestras → estimador más preciso.

    Args:
        v:         salida de la red, shape (batch, d)
        x:         input con requires_grad=True, shape (batch, d)
        n_samples: número de vectores aleatorios para promediar

    Returns:
        estimación de tr(∂v/∂x), shape (batch,)
    """
    trace_est = torch.zeros(x.shape[0], device=x.device)
    for _ in range(n_samples):
        eps    = torch.randn_like(x)
        Jt_eps = torch.autograd.grad(
            (eps * v).sum(), x,
            create_graph=True, retain_graph=True
        )[0]
        trace_est += (Jt_eps * eps).sum(dim=-1)
    return trace_est / n_samples