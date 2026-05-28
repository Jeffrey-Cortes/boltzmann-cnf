"""
analisis/plots.py
=================
Funciones de visualización para los tres sistemas de prueba.

Para agregar plots de un nuevo sistema:
    1. Define plot_nuevo_sistema() aquí
    2. Agrégalo al dispatcher plot_por_sistema()
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ─── Plots genéricos ──────────────────────────────────────────────────────────

def plot_training_curve(losses, run_dir):
    """Pérdida cruda + media móvil de 100 pasos."""
    losses = np.array(losses)
    window = 100
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(9, 3))
    plt.plot(losses, lw=0.5, color="steelblue", alpha=0.4, label="pérdida")
    plt.plot(range(window - 1, len(losses)), smooth,
             lw=1.5, color="steelblue", label=f"media móvil ({window})")
    plt.xlabel("Paso")
    plt.ylabel("Pérdida")
    plt.title("Curva de entrenamiento")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(run_dir, "training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    # print(f"Guardado: {path}")


def plot_comparacion(samples_net, samples_mcmc, beta, run_dir):
    """Histograma 2D de las dos primeras dimensiones: red vs MCMC."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(
        axes,
        [samples_net, samples_mcmc],
        ["Red neuronal (propuesta)", "MCMC (referencia)"],
    ):
        ax.hist2d(data[:, 0], data[:, 1], bins=60, cmap="plasma")
        ax.set(title=title, xlabel="x₁", ylabel="x₂",
               xlim=(-3, 3), ylim=(-3, 3))
    plt.suptitle(f"Distribución de Boltzmann (β={beta})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(run_dir, "comparacion.png")
    plt.savefig(path, dpi=150)
    plt.close()
    # print(f"Guardado: {path}")


# ─── H_harmonic ───────────────────────────────────────────────────────────────

def plot_harmonic(samples_net, samples_mcmc, beta, run_dir):
    """
    Marginal por dimensión para el oscilador armónico.
    Predicción exacta: cada xᵢ ~ N(0, 1/β).
    """
    dim       = samples_net.shape[1]
    n_show    = min(dim, 4)
    sigma_teo = 1.0 / np.sqrt(beta)
    x_range   = np.linspace(-4 * sigma_teo, 4 * sigma_teo, 400)
    g_teo     = np.exp(-0.5 * x_range**2 / sigma_teo**2)
    g_teo    /= np.trapz(g_teo, x_range)

    fig, axes = plt.subplots(1, n_show, figsize=(4.5 * n_show, 4), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.hist(samples_net[:, i],  bins=60, density=True,
                alpha=0.6, label="Red",  color="steelblue")
        ax.hist(samples_mcmc[:, i], bins=60, density=True,
                alpha=0.5, label="MCMC", color="tomato")
        ax.plot(x_range, g_teo, "k--", lw=2,
                label=f"N(0,1/β)  σ={sigma_teo:.2f}")
        ax.set_title(f"Dimensión {i+1}")
        ax.set_xlabel(f"x{i+1}")
        if i == 0:
            ax.set_ylabel("densidad")
        ax.legend(fontsize=8)

    plt.suptitle(f"Oscilador armónico {dim}D — marginales (β={beta})",
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(run_dir, "marginales_harmonic.png")
    plt.savefig(path, dpi=150)
    plt.close()
    # print(f"Guardado: {path}")


# ─── H_double_well ────────────────────────────────────────────────────────────

def plot_double_well(samples_net, samples_mcmc, beta, run_dir):
    """
    Marginales para el doble pozo ND: H = Σᵢ (xᵢ²-1)²

    Muestra hasta 4 dimensiones. Todas las marginales son idénticas por
    simetría: p(xᵢ) ∝ exp(-β(xᵢ²-1)²), bimodal con picos en ±1.
    La curva teórica se calcula por integración numérica 1D.
    """
    dim    = samples_net.shape[1]
    n_show = min(dim, 4)

    # Marginal teórica exacta (idéntica para todas las dimensiones)
    x_range = np.linspace(-3, 3, 600)
    dx      = x_range[1] - x_range[0]
    p_teo   = np.exp(-beta * (x_range**2 - 1)**2)
    p_teo  /= p_teo.sum() * dx

    fig, axes = plt.subplots(1, n_show, figsize=(4.5 * n_show, 4), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.hist(samples_net[:, i],  bins=60, density=True,
                alpha=0.6, label="Red",  color="steelblue")
        ax.hist(samples_mcmc[:, i], bins=60, density=True,
                alpha=0.5, label="MCMC", color="tomato")
        ax.plot(x_range, p_teo, "k--", lw=2, label="Teórico")
        ax.set_title(f"Dimensión {i+1}")
        ax.set_xlabel(f"x{i+1}")
        if i == 0:
            ax.set_ylabel("densidad")
        ax.legend(fontsize=8)

    n_min = 2**min(dim, 10)
    suffix = "+" if dim > 10 else ""
    plt.suptitle(
        f"Doble pozo {dim}D — marginales (β={beta})\n"
        f"H = Σᵢ(xᵢ²−1)²,  {n_min}{suffix} mínimos en {{±1}}^{dim}",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(run_dir, "marginales_double_well.png")
    plt.savefig(path, dpi=150)
    plt.close()
    # print(f"Guardado: {path}")


# ─── Cadena de Debye 1D ───────────────────────────────────────────────────────

def plot_debye_chain_1D(samples_net, samples_mcmc, Sigma_exact, meta, beta, run_dir):
    """
    Cuatro paneles para la cadena de Debye 1D:
    1. C(r): correlaciones vs distancia — red vs MCMC vs exacto
    2. g(ω): densidad de estados — eigenvalores de Σ_emp vs exacto analítico
    3. Marginal del sitio central — red, MCMC y gaussiana exacta
    4. Curva de dispersión ω(k) — teórica vs estimada de muestras
    """
    N         = meta["N"]
    f         = meta["f"]
    omega_max = meta["omega_max"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ── Panel 1: C(r) ─────────────────────────────────────────────────────
    corr_net   = np.corrcoef(samples_net.T)
    corr_mcmc  = np.corrcoef(samples_mcmc.T)
    corr_exact = (Sigma_exact /
                  np.sqrt(np.outer(np.diag(Sigma_exact),
                                   np.diag(Sigma_exact))))

    ref    = N // 2
    dists  = np.array([min(abs(i - ref), N - abs(i - ref)) for i in range(N)])
    r_vals = np.arange(N // 2 + 1)
    c_net_r, c_mcmc_r, c_ex_r = [], [], []
    for r in r_vals:
        mask = dists == r
        c_net_r.append( corr_net[ref,   mask].mean())
        c_mcmc_r.append(corr_mcmc[ref,  mask].mean())
        c_ex_r.append(  corr_exact[ref, mask].mean())

    axes[0].plot(r_vals, c_ex_r,   'ko--', lw=2,   label='Exacta')
    axes[0].plot(r_vals, c_net_r,  'b^-',  lw=1.5, label='Red')
    axes[0].plot(r_vals, c_mcmc_r, 'r^-',  lw=1.5, label='MCMC')
    axes[0].set(xlabel='Distancia  r  (sitios)', ylabel='C(r)',
                title='Correlaciones de la cadena')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Panel 2: g(ω) ─────────────────────────────────────────────────────
    def freqs_from_cov(samples):
        eigvals = np.linalg.eigvalsh(np.cov(samples.T))
        eigvals = np.maximum(eigvals, 1e-10)
        return np.sqrt(1.0 / (beta * eigvals))

    freqs_emp  = freqs_from_cov(samples_net)
    freqs_mcmc = freqs_from_cov(samples_mcmc)

    bins = 35
    kw   = dict(density=True, alpha=0.5, bins=bins, range=(0, omega_max * 1.1))
    axes[1].hist(freqs_emp,  label='Red',  color='steelblue', **kw)
    axes[1].hist(freqs_mcmc, label='MCMC', color='tomato',    **kw)

    om     = np.linspace(1e-3, omega_max * 0.999, 500)
    g_real = 1.0 / (np.pi * np.sqrt(np.maximum(omega_max**2 - om**2, 1e-12)))
    g_real /= np.trapz(g_real, om)
    axes[1].plot(om, g_real, 'k-',  lw=2,   label='Cadena exacta [Ec. 11-47]')

    g_deb  = np.ones_like(om) / omega_max
    g_deb /= np.trapz(g_deb, om)
    axes[1].plot(om, g_deb,  'k--', lw=1.5, label='Aprox. Debye 1D')

    axes[1].set(xlabel='ω', ylabel='g(ω)',
                title='Densidad de estados')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: Marginal sitio central ───────────────────────────────────
    sigma_teo = np.sqrt(Sigma_exact[ref, ref])
    x_range   = np.linspace(-4 * sigma_teo, 4 * sigma_teo, 300)
    p_teo     = np.exp(-0.5 * x_range**2 / sigma_teo**2)
    p_teo    /= np.trapz(p_teo, x_range)

    axes[2].hist(samples_net[:, ref],  bins=60, density=True,
                 alpha=0.6, label='Red',  color='steelblue')
    axes[2].hist(samples_mcmc[:, ref], bins=60, density=True,
                 alpha=0.5, label='MCMC', color='tomato')
    axes[2].plot(x_range, p_teo, 'k--', lw=2,
                 label=f'Exacta  σ={sigma_teo:.3f}')
    axes[2].set(title=f'Marginal sitio central  j={ref}',
                xlabel='x', ylabel='densidad')
    axes[2].legend(fontsize=8)

    # ── Panel 4: Curva de dispersión ──────────────────────────────────────
    k_idx  = np.arange(N)
    om_teo = omega_max * np.abs(np.sin(np.pi * k_idx / N))

    eigvals_emp    = np.maximum(np.sort(np.linalg.eigvalsh(np.cov(samples_net.T))),  1e-10)
    eigvals_mcmc   = np.maximum(np.sort(np.linalg.eigvalsh(np.cov(samples_mcmc.T))), 1e-10)
    om_emp_sorted  = np.sort(np.sqrt(1.0 / (beta * eigvals_emp)))
    om_mcmc_sorted = np.sort(np.sqrt(1.0 / (beta * eigvals_mcmc)))

    k_plot = np.arange(N) / N
    axes[3].plot(k_plot, np.sort(om_teo),   'k-',  lw=2,   label='Teórica [Ec. 11-40]')
    axes[3].plot(k_plot, om_emp_sorted,      'b^',  ms=4,   label='Red',  alpha=0.7)
    axes[3].plot(k_plot, om_mcmc_sorted,     'ro',  ms=4,   label='MCMC', alpha=0.7)
    axes[3].plot(k_plot, omega_max * k_plot, 'k--', lw=1.5, label='Debye (lineal)')
    axes[3].set(xlabel='k / k_max  (zona de Brillouin)', ylabel='ω',
                title='Curva de dispersión [Ec. 11-40]')
    axes[3].legend(fontsize=7)
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(f'Cadena de Debye 1D — N={N},  f={f},  β={beta}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(run_dir, 'debye_chain_1D.png')
    plt.savefig(path, dpi=150)
    plt.close()
    # print(f'Guardado: {path}')


def plot_phi4_chain(samples_net, samples_mcmc, meta, beta, run_dir):
    """
    Cinco paneles para la cadena φ⁴ 1D:
    1. Histograma de m = (1/N) Σ xⱼ  — bimodalidad / colapso de modo
    2. C(r) = ⟨x₀ xᵣ⟩ - ⟨x₀⟩⟨xᵣ⟩  — correlaciones + ajuste exponencial ξ
    3. S(k) = (1/N)|Σⱼ xⱼ e^{2πijk/N}|²  — factor de estructura (eje y desde 0)
    4. Marginal del sitio central  — vs doble pozo desacoplado y gaussiana efectiva
    5. Scatter (x₀, x₁): correlación entre vecinos inmediatos
 
    Cambios respecto a la versión anterior:
    - Panel 1: rango de m ampliado a [-2, 2] para no cortar distribuciones
              con colapso de modo (N=32, f=1) ni la fase ordenada (f=5)
    - Panel 2: se añade ajuste exponencial C(r) ~ A·exp(-r/ξ) con anotación
              de ξ para red y MCMC — la figura es autosuficiente sin el texto
    - Panel 3: eje y empieza en 0 (antes empezaba en el mínimo de los datos,
              exagerando visualmente las diferencias relativas)
    """
 
    N = meta["N"]
    f = meta["f"]
 
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
 
    # ── Panel 1: parámetro de orden m ─────────────────────────────────────
    m_net  = samples_net.mean(axis=1)
    m_mcmc = samples_mcmc.mean(axis=1)
 
    # Rango [-2, 2]: cubre colapso de modo (red en ±0.9) y fase desordenada
    bins = np.linspace(-2.0, 2.0, 70)
    axes[0].hist(m_net,  bins=bins, density=True,
                 alpha=0.6, label="Red",  color="steelblue")
    axes[0].hist(m_mcmc, bins=bins, density=True,
                 alpha=0.5, label="MCMC", color="tomato")
    axes[0].axvline(0, color="k", lw=1, ls="--")
    axes[0].set(xlabel=r"$m = \frac{1}{N} \sum_j x_j$", ylabel="densidad",
                title=f"Parámetro de orden  ($f={f}$)")
    axes[0].legend(fontsize=8)
 
    # ── Panel 2: C(r) con ajuste exponencial ──────────────────────────────
    def corr_distancia(samples):
        """Correlación conectada normalizada, promediada sobre sitios."""
        mu    = samples.mean(axis=0)
        fluct = samples - mu[None, :]
        C = np.array([
            (fluct * np.roll(fluct, r, axis=1)).mean()
            for r in range(N // 2 + 1)
        ])
        return C / C[0]
 
    def ajuste_xi(C):
        """
        Ajuste lineal de log|C(r)| vs r para r >= 1.
        Devuelve ξ = -1/pendiente si la pendiente es negativa, inf si no.
        """
        r_fit = np.arange(1, N // 2 + 1)
        C_fit = np.abs(C[1:]) + 1e-12
        try:
            coef = np.polyfit(r_fit, np.log(C_fit), 1)
            xi = -1.0 / coef[0] if coef[0] < 0 else np.inf
            return xi, coef
        except Exception:
            return np.nan, None
 
    r_vals = np.arange(N // 2 + 1)
    C_net  = corr_distancia(samples_net)
    C_mcmc = corr_distancia(samples_mcmc)
 
    xi_net,  coef_net  = ajuste_xi(C_net)
    xi_mcmc, coef_mcmc = ajuste_xi(C_mcmc)
 
    axes[1].plot(r_vals, C_mcmc, "ro-",  ms=5, lw=1.5, label="MCMC")
    axes[1].plot(r_vals, C_net,  "b^--", ms=5, lw=1.5, label="Red")
 
    # Superponer ajuste exponencial (solo si el ajuste es válido)
    r_cont = np.linspace(0, N // 2, 200)
    if coef_mcmc is not None and np.isfinite(xi_mcmc):
        axes[1].plot(r_cont,
                     np.exp(-r_cont / xi_mcmc),
                     "r-", lw=1, alpha=0.5,
                     label=rf"$\xi_\mathrm{{MCMC}}={xi_mcmc:.2f}$")
    if coef_net is not None and np.isfinite(xi_net):
        axes[1].plot(r_cont,
                     np.exp(-r_cont / xi_net),
                     "b-", lw=1, alpha=0.5,
                     label=rf"$\xi_\mathrm{{red}}={xi_net:.2f}$")
 
    axes[1].axhline(0, color="k", lw=0.8, ls=":")
    axes[1].set(xlabel="$r$  (sitios)", ylabel="$C(r) / C(0)$",
                title="Correlación conectada")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
 
    # ── Panel 3: S(k) factor de estructura — eje y desde 0 ───────────────
    def factor_estructura(samples):
        """S(k) = (1/N) ⟨|FFT(x)|²⟩, promedio sobre muestras."""
        N_s   = samples.shape[1]
        fft   = np.fft.fft(samples, axis=1)
        Sk    = (np.abs(fft) ** 2).mean(axis=0) / N_s
        freqs = np.arange(N_s) / N_s
        return freqs, Sk
 
    k_net,  Sk_net  = factor_estructura(samples_net)
    k_mcmc, Sk_mcmc = factor_estructura(samples_mcmc)
 
    half = N // 2
    axes[2].plot(k_mcmc[:half], Sk_mcmc[:half], "ro-",  ms=4, lw=1.5,
                 label=f"MCMC  $S(0)={Sk_mcmc[0]:.2f}$")
    axes[2].plot(k_net[:half],  Sk_net[:half],  "b^--", ms=4, lw=1.5,
                 label=f"Red   $S(0)={Sk_net[0]:.2f}$")
    # eje y desde 0: evita exagerar diferencias relativas pequeñas
    axes[2].set_ylim(bottom=0)
    axes[2].set(xlabel="$k / k_\mathrm{max}$", ylabel="$S(k)$",
                title="Factor de estructura")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)
 
    # ── Panel 4: marginal sitio central ───────────────────────────────────
    ref       = N // 2
    sigma_eff = samples_mcmc[:, ref].std()
    mu_eff    = samples_mcmc[:, ref].mean()
    x_range   = np.linspace(mu_eff - 4*sigma_eff, mu_eff + 4*sigma_eff, 300)
 
    p_gauss   = np.exp(-0.5 * ((x_range - mu_eff) / sigma_eff)**2)
    p_gauss  /= np.trapz(p_gauss, x_range)
 
    p_dw      = np.exp(-beta * (x_range**2 - 1)**2)
    p_dw     /= np.trapz(p_dw, x_range)
 
    axes[3].hist(samples_net[:, ref],  bins=60, density=True,
                 alpha=0.6, label="Red",  color="steelblue")
    axes[3].hist(samples_mcmc[:, ref], bins=60, density=True,
                 alpha=0.5, label="MCMC", color="tomato")
    axes[3].plot(x_range, p_dw,    "k--", lw=2, label="Doble pozo ($f=0$)")
    axes[3].plot(x_range, p_gauss, "k:",  lw=2,
                 label=rf"Gauss ef. $\sigma={sigma_eff:.2f}$")
    axes[3].set(title=f"Marginal sitio $j={ref}$",
                xlabel="$x$", ylabel="densidad")
    axes[3].legend(fontsize=7)
 
    # ── Panel 5: scatter (x₀, x₁) — correlación entre vecinos ───────────
    n_show = min(1000, samples_net.shape[0])
    axes[4].scatter(samples_mcmc[:n_show, 0], samples_mcmc[:n_show, 1],
                    s=2, alpha=0.3, color="tomato", label="MCMC")
    axes[4].scatter(samples_net[:n_show, 0],  samples_net[:n_show, 1],
                    s=2, alpha=0.3, color="steelblue", label="Red")
    axes[4].set(xlabel="$x_0$", ylabel="$x_1$",
                title="Correlación entre vecinos")
    axes[4].legend(fontsize=8, markerscale=4)
 
    plt.suptitle(
        rf"Cadena $\varphi^4$ 1D — $N={N}$,  $f={f}$,  $\beta={beta}$"
        f"\n({meta['regimen']})",
        fontsize=12
    )
    plt.tight_layout()
    path = os.path.join(run_dir, "phi4_chain.png")
    plt.savefig(path, dpi=150)
    plt.close()
# ─── Dispatcher ───────────────────────────────────────────────────────────────

def plot_por_sistema(samples_net, samples_mcmc, beta, hamiltoniano,
                     run_dir, **kwargs):
    """
    Llama al plot específico según el hamiltoniano.
    kwargs permite pasar Sigma_exact y meta para sistemas con solución exacta.
    """
    dispatch = {
        "double_well"    : lambda: plot_double_well(
                               samples_net, samples_mcmc, beta, run_dir),
        "harmonic"       : lambda: plot_harmonic(
                               samples_net, samples_mcmc, beta, run_dir),
        "debye_chain_1D" : lambda: plot_debye_chain_1D(
                               samples_net, samples_mcmc,
                               kwargs["Sigma_exact"],
                               kwargs["meta"],
                               beta, run_dir),
        "phi4_chain" : lambda: plot_phi4_chain(
                   samples_net, samples_mcmc,
                   kwargs["meta"], beta, run_dir),
    }
    if hamiltoniano not in dispatch:
        raise ValueError(f"Hamiltoniano desconocido: '{hamiltoniano}'. "
                         f"Disponibles: {list(dispatch.keys())}")
    dispatch[hamiltoniano]()