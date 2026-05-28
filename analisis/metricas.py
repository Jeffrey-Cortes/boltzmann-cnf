"""
analisis/metricas.py
====================
Métricas cuantitativas para evaluar la calidad del muestreo.

Funciones principales:
    calcular_metricas()   — runner que llama a todas las métricas
    metrica_covarianza()  — error de Frobenius vs solución exacta
    calor_especifico()    — observable termodinámico comparable con libros

Métricas implementadas:
    1. Energía media E[H(x)]
    2. MMD  (Maximum Mean Discrepancy)
    3. KL por KDE  (solo 2D — harmonic y double_well)
    4. Error de covarianza  (Frobenius, varianzas, correlaciones)
    5. Calor específico Cv = β² Var(H)
"""

import os
import numpy as np
import torch
from scipy.stats import gaussian_kde


# ─── 1. Energía media ─────────────────────────────────────────────────────────

def energia_media(samples, H_func):
    """
    E[H(x)] sobre un conjunto de muestras.
    La distribución correcta satisface E_π[H] = -∂/∂β log Z.

    Returns:
        (media, desviación estándar)
    """
    x      = torch.tensor(samples, dtype=torch.float32)
    H_vals = H_func(x).detach().numpy()
    return H_vals.mean(), H_vals.std()


# ─── 2. MMD ───────────────────────────────────────────────────────────────────

def mmd(samples_p, samples_q, sigma=None):
    """
    Maximum Mean Discrepancy con kernel gaussiano.

        MMD²(P,Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    Funciona en cualquier dimensión sin estimar densidades.
    MMD ≈ 0 → distribuciones similares. Valores < 0.05 indican buena concordancia.

    Referencia: Gretton et al. (2012), "A Kernel Two-Sample Test".
    """
    p = torch.tensor(samples_p, dtype=torch.float32)
    q = torch.tensor(samples_q, dtype=torch.float32)

    if sigma is None:
        all_samples = torch.cat([p, q], dim=0)
        dists       = torch.cdist(all_samples, all_samples)
        sigma       = dists.median().item() + 1e-6

    def kernel(a, b):
        return torch.exp(-torch.cdist(a, b)**2 / (2 * sigma**2))

    mmd2 = kernel(p, p).mean() + kernel(q, q).mean() - 2 * kernel(p, q).mean()
    return float(mmd2.clamp(min=0).sqrt())


# ─── 3. KL por KDE ────────────────────────────────────────────────────────────

def kl_kde(samples_p, samples_q, n_eval=1000):
    """
    Estima KL(P||Q) usando KDE en 2D.
    Solo confiable para dim=2 — en dimensión alta la KDE colapsa.
    Se usa en: harmonic (dim=2), double_well (dim=2).
    """
    if samples_p.shape[1] != 2:
        return None

    kde_p = gaussian_kde(samples_p.T)
    kde_q = gaussian_kde(samples_q.T)

    x_min = min(samples_p[:, 0].min(), samples_q[:, 0].min()) - 0.5
    x_max = max(samples_p[:, 0].max(), samples_q[:, 0].max()) + 0.5
    y_min = min(samples_p[:, 1].min(), samples_q[:, 1].min()) - 0.5
    y_max = max(samples_p[:, 1].max(), samples_q[:, 1].max()) + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_eval),
                         np.linspace(y_min, y_max, n_eval))
    puntos = np.vstack([xx.ravel(), yy.ravel()])
    p_vals = kde_p(puntos).reshape(n_eval, n_eval)
    q_vals = kde_q(puntos).reshape(n_eval, n_eval)

    mask = (p_vals > 1e-10) & (q_vals > 1e-10)
    dx   = (x_max - x_min) / n_eval
    dy   = (y_max - y_min) / n_eval
    kl   = (p_vals[mask] * np.log(p_vals[mask] / q_vals[mask])).sum() * dx * dy
    return float(kl)


# ─── 4. Error de covarianza ───────────────────────────────────────────────────

def metrica_covarianza(samples, Sigma_exact, run_dir, label="red"):
    """
    Compara la covarianza empírica con la solución exacta.

    Reporta:
        - Error de Frobenius relativo: ||Σ_emp - Σ_exact||_F / ||Σ_exact||_F
        - Error en varianzas: error relativo medio en la diagonal
        - Error en correlaciones: error absoluto medio en la matriz de correlación

    Se usa en: harmonic (Sigma_exact = I/β), debye_chain_1D (Sigma_exact = (βK)⁻¹).
    """
    import matplotlib.pyplot as plt

    Sigma_emp = np.cov(samples.T)
    frob_rel  = (np.linalg.norm(Sigma_emp - Sigma_exact, "fro")
                 / np.linalg.norm(Sigma_exact, "fro"))

    var_exact = np.diag(Sigma_exact)
    err_var   = np.abs(np.diag(Sigma_emp) - var_exact).mean() / var_exact.mean()

    def to_corr(S):
        d_inv = 1.0 / np.sqrt(np.diag(S))
        return S * np.outer(d_inv, d_inv)

    err_corr = np.abs(to_corr(Sigma_emp) - to_corr(Sigma_exact)).mean()

    print(f"\n  [{label}] Frobenius relativo : {frob_rel:.4f}")
    print(f"  [{label}] Error varianzas     : {err_var:.4f}")
    print(f"  [{label}] Error correlaciones : {err_corr:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmax = np.abs(Sigma_exact).max()
    diff = Sigma_emp - Sigma_exact

    for ax, mat, title, vmin in zip(
        axes,
        [Sigma_exact, Sigma_emp, diff],
        ["Σ exacta", f"Σ empírica ({label})",
         f"Diferencia (Frob. rel.={frob_rel:.3f})"],
        [-vmax, -vmax, -np.abs(diff).max()],
    ):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=-vmin)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"Covarianza — {label}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(run_dir, f"covarianza_{label}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")

    return frob_rel, err_var, err_corr


# ─── 5. Calor específico ──────────────────────────────────────────────────────

def calor_especifico(samples, H_func, beta, label="red"):
    """
    Calor específico: Cv = β² Var(H).

    Observable termodinámico estándar comparable con libros de texto.
    Valores teóricos exactos:
        - Armónico ND:    Cv = d/2
        - Doble pozo 2D:  pico en β ≈ 2.4 (anomalía de Schottky)
        - Debye 1D:       Cv = N/2 (equipartición clásica)

    Referencia: Pathria & Beale, Statistical Mechanics, Cap. 3.
    """
    x      = torch.tensor(samples, dtype=torch.float32)
    H_vals = H_func(x).detach().numpy()

    E_mean  = H_vals.mean()
    E2_mean = (H_vals**2).mean()
    Cv      = beta**2 * (E2_mean - E_mean**2)

    print(f"\n  [{label}] Energía media   ⟨H⟩ = {E_mean:.4f}")
    print(f"  [{label}] Calor específico Cv  = {Cv:.4f}")

    return E_mean, Cv

def valores_teoricos_debye_chain_1D(N, beta):
    """
    Valores teóricos exactos de la cadena de Debye 1D con PBC.

    Por equipartición clásica, cada uno de los N modos normales contribuye
    1/(2β) a la energía y 1/2 al calor específico:

        <H> = N / (2*beta)
        Cv  = N / 2

    Args:
        N:    numero de atomos (dimension del sistema)
        beta: inverso de temperatura

    Returns:
        (E_mean, Cv) — energia media y calor especifico exactos
    """
    E_mean = N / (2.0 * beta)
    Cv     = N / 2.0
    return E_mean, Cv


def valores_teoricos_double_well_nd(dim, beta):
    x = np.linspace(-3, 3, 10000)
    p = np.exp(-beta * (x**2 - 1)**2)
    p /= np.trapz(p, x)

    # Por simetría, cada dimensión es idéntica
    E1    = np.trapz((x**2 - 1)**2 * p, x)       # ⟨(xi²-1)²⟩ por dim
    E2    = np.trapz((x**2 - 1)**4 * p, x)       # ⟨(xi²-1)⁴⟩ por dim
    var1  = E2 - E1**2                             # Var(Hi) por dim

    E_total  = dim * E1                            # ⟨H⟩ total
    Cv_total = beta**2 * dim * var1                # Cv total (dims independientes)

    return E_total, Cv_total

def valores_teoricos_harmonic_nd(dim, beta):
    """
    Valores teóricos exactos del oscilador armónico ND: H = ‖x‖²/2

    Derivación analítica (equipartición clásica):
        xᵢ ~ N(0, 1/β)  →  ⟨xᵢ²⟩ = 1/β,  Var(xᵢ²) = 2/β²

        ⟨H⟩ = d / (2β)
        Cᵥ  = β² Var(H) = d / 2   (independiente de β)

    Args:
        dim:  número de dimensiones
        beta: inverso de temperatura

    Returns:
        (E_mean, Cv)  — energía media y calor específico exactos
    """
    E_mean = dim / (2.0 * beta)
    Cv     = dim / 2.0
    return E_mean, Cv

def observables_phi4(samples_net, samples_mcmc, meta, beta, run_dir):
    """
    Observables específicos de la cadena φ⁴.
 
    Calcula y guarda en observables_phi4.txt:
      1. ⟨|m|⟩  — parámetro de orden
      2. ⟨m²⟩   — segundo momento de la magnetización
      3. χ = Nβ(⟨m²⟩ - ⟨|m|⟩²)  — susceptibilidad magnética
      4. ξ       — longitud de correlación (ajuste exp. a C(r))
      5. S(k=0)  — factor de estructura en k=0: S(0) = N·⟨m²⟩
 
    Cambios respecto a la versión anterior:
    - Se añade S(k=0) = N·⟨m²⟩: es el observable que más claramente
      distingue colapso de modo (S_red ≈ 0) de distribución correcta
      (S_MCMC ≈ 12 en régimen crítico). Estaba en la tabla de resultados
      pero no se imprimía ni se guardaba.
    - Se corrige la lógica de lineas: cada iteración del loop imprime
      solo sus propias líneas, sin arrastrar la última línea del anterior.
    """
 
    N = meta["N"]
 
    encabezado = ["\n=== Observables φ⁴ ==="]
 
    bloques = []   # lista de bloques de texto, uno por fuente
 
    for label, samples in [("red", samples_net), ("mcmc", samples_mcmc)]:
        m     = samples.mean(axis=1)      # (n,)  magnetización por muestra
        m_abs = np.abs(m).mean()          # ⟨|m|⟩
        m2    = (m**2).mean()             # ⟨m²⟩
        chi   = N * beta * (m2 - m_abs**2)  # susceptibilidad χ
 
        # S(k=0) = N·⟨m²⟩  (identidad exacta por definición de S(k))
        Sk0 = N * m2
 
        # Longitud de correlación: ajuste C(r) ~ exp(-r/ξ) para r ≥ 1
        mu    = samples.mean(axis=0)
        fluct = samples - mu[None, :]
        C = np.array([
            (fluct * np.roll(fluct, r, axis=1)).mean()
            for r in range(N // 2 + 1)
        ])
        C /= C[0]
        r_fit = np.arange(1, N // 2 + 1)
        C_fit = np.abs(C[1:]) + 1e-12
        try:
            coef = np.polyfit(r_fit, np.log(C_fit), 1)
            xi   = -1.0 / coef[0] if coef[0] < 0 else float("inf")
        except Exception:
            xi = float("nan")
 
        bloque = [
            f"\n  [{label}]",
            f"    ⟨|m|⟩              = {m_abs:.4f}",
            f"    ⟨m²⟩               = {m2:.4f}",
            f"    χ = Nβ(⟨m²⟩-⟨|m|⟩²) = {chi:.4f}",
            f"    ξ                  ≈ {xi:.3f}  sitios",
            f"    S(k=0) = N·⟨m²⟩   = {Sk0:.4f}",
        ]
        bloques.append(bloque)
        # Imprimir solo este bloque (sin contaminar con el anterior)
        print("\n".join(bloque))
 
    # Escribir todo en el archivo de texto
    todas_las_lineas = encabezado
    for b in bloques:
        todas_las_lineas += b
 
    with open(os.path.join(run_dir, "observables_phi4.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(todas_las_lineas))
# ─── Runner principal ─────────────────────────────────────────────────────────

def calcular_metricas(samples_net, samples_mcmc, H_func, run_dir, beta=None):
    """
    Calcula y reporta todas las métricas. Guarda metricas.txt en run_dir.

    Args:
        samples_net:  muestras de la red, shape (n, d)
        samples_mcmc: muestras de MCMC, shape (n, d)
        H_func:       hamiltoniano
        run_dir:      carpeta de la corrida
        beta:         si se provee, también calcula Cv
    """
    lineas = ["=== Métricas cuantitativas ===\n"]

    # Energía media
    mu_net,  std_net  = energia_media(samples_net,  H_func)
    mu_mcmc, std_mcmc = energia_media(samples_mcmc, H_func)
    lineas += [
        "Energía media  E[H(x)]:",
        f"  Red:  {mu_net:.4f} ± {std_net:.4f}",
        f"  MCMC: {mu_mcmc:.4f} ± {std_mcmc:.4f}",
        f"  Δ(red-MCMC): {abs(mu_net - mu_mcmc):.4f}\n",
    ]

    # MMD
    mmd_val = mmd(samples_net, samples_mcmc)
    lineas += [
        f"MMD(red, MCMC): {mmd_val:.6f}",
        "  (ideal ≈ 0; valores < 0.05 indican buena concordancia)\n",
    ]

    # KL por KDE (solo 2D)
    if samples_net.shape[1] == 2:
        kl_val = kl_kde(samples_net, samples_mcmc)
        lineas += [
            f"KL_KDE(red || MCMC): {kl_val:.4f}",
            "  (ideal = 0; se estima sobre malla 2D)\n",
        ]
    else:
        lineas.append("KL_KDE: omitida (solo confiable en 2D)\n")

    # Calor específico
    if beta is not None:
        _, Cv_net  = calor_especifico(samples_net,  H_func, beta, label="red")
        _, Cv_mcmc = calor_especifico(samples_mcmc, H_func, beta, label="mcmc")
        lineas += [
            f"\nCalor específico Cv = β² Var(H):",
            f"  Red:  {Cv_net:.4f}",
            f"  MCMC: {Cv_mcmc:.4f}\n",
        ]

    for l in lineas:
        print(l)

    with open(os.path.join(run_dir, "metricas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))