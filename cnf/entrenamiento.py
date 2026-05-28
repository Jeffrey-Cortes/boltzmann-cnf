"""
cnf/entrenamiento.py
====================
Función de pérdida y bucle de entrenamiento.

La pérdida implementa:
    L(θ) = E_{x₀~π₀} [ β·H(x₁(θ)) − ∫₀¹ tr(∂v_θ/∂x_t) dt ]

donde la integral se discretiza con K pasos de Euler y la traza
se estima con el estimador de Hutchinson.
"""

import torch
from cnf.modelo import VelocityNet, hutchinson_trace


def loss_fn(net, x0, H_func, beta, K=4, n_hutchinson=3):
    """
    Calcula L(θ) con K pasos de Euler.

    Cada paso:
        x ← x + (1/K) · v_θ(x)
        traza acumulada += (1/K) · tr(∂v_θ/∂x)

    Args:
        net:          VelocityNet
        x0:           muestras iniciales de π₀, shape (batch, d)
        H_func:       hamiltoniano H(x), devuelve shape (batch,)
        beta:         inverso de temperatura
        K:            pasos de Euler (más pasos = integración más fiel)
        n_hutchinson: vectores para estimar la traza (más = menos varianza)

    Returns:
        pérdida escalar
    """
    x = x0
    trace_total = 0.0
    for _ in range(K):
        x = x.requires_grad_(True)
        v = net(x)
        trace_total = trace_total + hutchinson_trace(v, x, n_hutchinson) / K
        x = x + v / K
    return (beta * H_func(x) - trace_total).mean()


def train(H_func, dim=2, beta=2.0, n_steps=3000, batch_size=512,
          lr=1e-3, hidden=128, n_layers=3, n_hutchinson=3, K=4,
          log_every=500, clip_grad=None):
    """
    Bucle de entrenamiento del CNF.

    Nota sobre el scheduler: se probó CosineAnnealingLR y no produjo mejora
    estadísticamente significativa en ningún sistema. El error residual es
    atribuible al sesgo de KL(q||pi_H), no a la calidad de la optimización.
    Se mantiene lr fijo para simplicidad y reproducibilidad.

    Args:
        H_func:       hamiltoniano del sistema
        dim:          dimensión del espacio de configuraciones
        beta:         inverso de temperatura
        n_steps:      número de pasos de entrenamiento
        batch_size:   muestras por paso
        lr:           tasa de aprendizaje (fijo durante todo el entrenamiento)
        hidden:       neuronas por capa oculta
        n_layers:     número de capas ocultas
        n_hutchinson: vectores de Hutchinson por paso
        K:            pasos de Euler por paso de entrenamiento
        log_every:    frecuencia de reporte en consola
        clip_grad:    umbral de clipping de gradientes (None = sin clipping)

    Returns:
        (net, losses) — red entrenada y lista de pérdidas
    """
    net       = VelocityNet(dim, hidden, n_layers)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    losses    = []

    for step in range(n_steps + 1):
        x0   = torch.randn(batch_size, dim)
        loss = loss_fn(net, x0, H_func, beta, K=K, n_hutchinson=n_hutchinson)

        if torch.isnan(loss):
            print(f"  [aviso] NaN en paso {step}, saltando")
            continue

        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        optimizer.step()
        losses.append(loss.item())

        if step % log_every == 0:
            print(f"Paso {step:5d} | Pérdida: {loss.item():.4f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    return net, losses