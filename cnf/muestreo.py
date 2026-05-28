"""
cnf/muestreo.py
===============
Muestreo con la red entrenada y referencia MCMC.
"""

import torch
import numpy as np


def sample(net, n=5000, dim=2, K=4):
    """
    Genera n muestras de π_H aplicando K pasos de Euler con la red.

    Args:
        net: VelocityNet entrenada
        n:   número de muestras
        dim: dimensión
        K:   pasos de Euler (debe coincidir con el usado en entrenamiento)

    Returns:
        np.ndarray shape (n, dim)
    """
    net.eval()
    with torch.no_grad():
        x = torch.randn(n, dim)
        for _ in range(K):
            x = x + net(x) / K
    return x.numpy()


def ground_truth_mcmc(H_func, beta, n=5000, dim=2,
                      steps=200_000, sigma=None, Sigma_exact=None, thin=10,
                      adapt=True, adapt_every=500, target_rate=0.23):
    """
    Metropolis-Hastings para obtener muestras de referencia exactas,
    con ajuste adaptativo de sigma durante el burn-in.

    El paso sigma inicial se elige con la regla de Roberts et al. (1997):
        sigma = 2.38/sqrt(d), ajustada por la varianza real del target si
        Sigma_exact está disponible.

    Ajuste adaptativo (adapt=True):
        Cada `adapt_every` pasos durante el burn-in se evalúa la tasa de
        aceptación local y se ajusta sigma:
            tasa > target_rate  ->  sigma *= 1.1  (propuestas más grandes)
            tasa < target_rate  ->  sigma *= 0.9  (propuestas más chicas)
        Esto garantiza exploración correcta independientemente del sistema
        o la dimensión. Solo se adapta durante el burn-in; después es fijo.

    Referencia: Haario et al. (2001), "An adaptive Metropolis algorithm".

    Thinning: se guarda una muestra cada `thin` pasos tras el burn-in para
    reducir la autocorrelación. El total de pasos se ajusta automáticamente
    si steps < burn_in + n*thin.

    Args:
        H_func:       hamiltoniano
        beta:         inverso de temperatura
        n:            muestras a colectar (después del burn-in)
        dim:          dimensión
        steps:        pasos mínimos totales de la cadena
        sigma:        paso de propuesta inicial (None = automático)
        Sigma_exact:  covarianza exacta del target (mejora sigma inicial)
        thin:         factor de submuestreo — guarda 1 de cada `thin` pasos
        adapt:        si True, ajusta sigma durante el burn-in
        adapt_every:  cada cuántos pasos revisar y ajustar sigma
        target_rate:  tasa de aceptación objetivo (0.23 óptimo para dim alta)

    Returns:
        np.ndarray shape (n, dim)
    """
    if sigma is None:
        if Sigma_exact is not None:
            sigma = 2.38 * np.sqrt(np.diag(Sigma_exact).mean()) / np.sqrt(dim)
        else:
            sigma = 2.38 / np.sqrt(dim)

    burn_in      = steps // 5
    steps_needed = burn_in + n * thin
    if steps_needed > steps:
        print(f"  [aviso] steps ajustado: {steps} -> {steps_needed} "
              f"(burn_in={burn_in}, n={n}, thin={thin})")
        steps = steps_needed

    print(f"  MCMC sigma_inicial={sigma:.4f} | thin={thin} | "
          f"adapt={adapt} | burn_in={burn_in:,} | pasos_totales={steps:,}")

    x           = torch.zeros(dim)
    samples     = []
    n_accepted  = 0
    acc_ventana = 0   # aceptaciones en la ventana actual de adaptación

    for i in range(steps):
        x_prop    = x + sigma * torch.randn(dim)
        log_ratio = -beta * (H_func(x_prop.unsqueeze(0))
                             - H_func(x.unsqueeze(0)))
        acepto = torch.log(torch.rand(1)) < log_ratio
        if acepto:
            x           = x_prop
            n_accepted += 1
            acc_ventana += 1

        # Adaptacion de sigma durante el burn-in
        if adapt and i < burn_in and (i + 1) % adapt_every == 0:
            tasa_local = acc_ventana / adapt_every
            sigma *= 1.1 if tasa_local > target_rate else 0.9
            acc_ventana = 0

        # Colectar muestras tras el burn-in con thinning
        if i >= burn_in and (i - burn_in) % thin == 0 and len(samples) < n:
            samples.append(x.clone())

    tasa_global = n_accepted / steps
    print(f"  MCMC sigma_final={sigma:.4f} | "
          f"tasa aceptacion: {tasa_global:.3f}  (ideal {target_rate})")
    return torch.stack(samples).numpy()