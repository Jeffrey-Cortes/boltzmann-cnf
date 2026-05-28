# Boltzmann CNF

Flujo normalizante continuo (CNF) con campo de velocidades estático para muestrear
distribuciones de Boltzmann $\pi_H(x) \propto e^{-\beta H(x)}$ a partir únicamente
del hamiltoniano, sin muestras de referencia ni MCMC durante el entrenamiento.

Trabajo de licenciatura en Física — FCFM, BUAP.
Autor: Jeffrey Cortes Cantoran.

---

## Método

El campo de velocidades $v_\theta(x)$ (sin dependencia en $t$) transporta muestras
de $\mathcal{N}(0, I)$ hacia $\pi_H$. La función de pérdida minimiza
$D_\mathrm{KL}(q_\theta \| \pi_H)$ usando solo evaluaciones analíticas de $H$:

$$\mathcal{L}(\theta) = \mathbb{E}_{x_0 \sim \pi_0}\!\left[\beta H(x_1(\theta)) - \int_0^1 \mathrm{tr}\!\left(\frac{\partial v_\theta}{\partial x_t}\right)dt\right]$$

La integral se discretiza con $K$ pasos de Euler y la traza del jacobiano
se estima con el estimador de Hutchinson.

---

## Sistemas implementados

| Sistema | Hamiltoniano | Solución exacta |
|---|---|---|
| Oscilador armónico ND | $H = \|\mathbf{x}\|^2/2$ | $\mathcal{N}(0, \beta^{-1}I)$ |
| Doble pozo ND | $H = \sum_i (x_i^2-1)^2$ | No (integración numérica 1D) |
| Cadena de Debye 1D (PBC) | $H = \frac{f}{2}\sum_j(x_j-x_{j+1})^2$ | $\mathcal{N}(0,(\beta K)^{-1})$ |
| Cadena $\varphi^4$ 1D (PBC) | $H = \sum_j\left[(x_j^2-1)^2 + \frac{f}{2}(x_j-x_{j+1})^2\right]$ | No |

---

## Estructura

```
.
├── main.py                  # Punto de entrada y configuración
├── cnf/
│   ├── modelo.py            # VelocityNet (MLP con SiLU) y estimador de Hutchinson
│   ├── entrenamiento.py     # Función de pérdida y bucle de entrenamiento
│   └── muestreo.py          # Muestreo CNF y referencia MCMC adaptativo
├── fisica/
│   ├── hamiltonianos.py     # Oscilador armónico y doble pozo
│   └── cristales.py         # Cadenas de Debye y φ⁴ con PBC
├── analisis/
│   ├── metricas.py          # MMD, KL-KDE, Frobenius, Cv, observables φ⁴
│   └── plots.py             # Figuras por sistema y curva de entrenamiento
└── utils/
    └── logger.py            # Logger, guardado de configuración y carpetas de corrida
```

---

## Instalación

```bash
git clone https://github.com/Jeffrey-Cortes/boltzmann-cnf.git
cd boltzmann-cnf
pip install torch numpy scipy matplotlib
```

No se requiere GPU. Los experimentos de la tesis corrieron en CPU.

---

## Uso

Edita `CONFIG` en `main.py` para seleccionar el sistema y los hiperparámetros:

```python
CONFIG = dict(
    hamiltoniano = "phi4_chain",   # harmonic | double_well | debye_chain_1D | phi4_chain
    dim          = 16,             # dimensión / número de sitios
    k            = 1.0,            # acoplamiento f (cadenas)
    beta         = 2.0,            # inverso de temperatura
    K            = 10,             # pasos de Euler
    n_steps      = 10_000,         # pasos de entrenamiento
    ...
)
```

Luego:

```bash
python main.py
```

Los resultados (figuras, métricas, log, modelo) se guardan automáticamente en
`resultados/<sistema>_<timestamp>/`.

---

## Hiperparámetros estándar de la tesis

| Parámetro | Valor |
|---|---|
| $K$ (pasos de Euler) | 10 |
| Neuronas por capa | 128 |
| Capas ocultas | 3 |
| Learning rate | $3\times10^{-4}$ |
| Batch size | 512 |
| Pasos de entrenamiento | 10 000 |
| Vectores de Hutchinson | 4 |
| $\beta$ | 2.0 |

---

## Referencia

Cortes Cantoran, J. (2026). *Flujos normalizantes continuos para el muestreo
de distribuciones de Boltzmann*. Tesis de licenciatura, Facultad de Ciencias
Físico-Matemáticas, BUAP.
