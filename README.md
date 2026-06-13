# polyfit

Determine copolymerization reactivity ratios from experimental composition data.
polyfit fits the Mayo-Lewis equation (low conversion) or the Skeist integrated
equation (high conversion) to your measurements using maximum-likelihood estimation,
correctly propagating measurement uncertainties in **all** variables via the
Errors-in-Variables (EVM) approach.
Credible regions are derived from the full 2D posterior — no linearisation required.

**Reference:** R. Reischke, *Macromolecular Theory and Simulations* **32** (2023) 2200063.
DOI: [10.1002/mats.202200063](https://onlinelibrary.wiley.com/doi/10.1002/mats.202200063)

---

## Which model should I use?

| Fractional conversion | Model | Config key |
|---|---|---|
| X < ~5 % | Mayo-Lewis (instantaneous) | `high_conversion = false` |
| X > ~5 % | Skeist (integrated) | `high_conversion = true` |

At low conversion the feed composition barely changes, so the instantaneous
Mayo-Lewis equation is a good approximation.
At higher conversion the more reactive monomer is consumed faster and the feed
drifts — the Skeist model integrates Mayo-Lewis over the full conversion path
and returns the **cumulative** copolymer composition.

---

## Getting started

**Just want to run a fit?** Follow Option 1 — it is all you need.
Options 2 and 3 are for users who want to work in Python or automate runs from a terminal.

---

## Option 1 — GUI (click install)

The easiest route: a desktop application with no command-line or programming required.

### Download

**With Git:**
```bash
git clone https://github.com/rreischke/polyfit.git
```

**Without Git:** go to [github.com/rreischke/polyfit](https://github.com/rreischke/polyfit),
click **Code → Download ZIP**, and unzip.

### Install

Open the polyfit folder and run the installer for your platform.

| Platform | Action |
|---|---|
| macOS | Double-click `install.command` (right-click → Open if Gatekeeper blocks it) |
| Windows | Double-click `install.bat` |
| Linux | `bash install.sh` in a terminal |

The installer installs all Python dependencies and creates a desktop/application launcher.

> **Python required.** Check you have it: `python --version`.
> If not, download from [python.org/downloads](https://www.python.org/downloads/)
> and on Windows tick **"Add Python to PATH"** during installation.

### Launch

| Platform | How to launch |
|---|---|
| macOS | Double-click **polyfit** in Applications |
| Windows | Double-click the **polyfit** shortcut on the Desktop |
| Linux | Search for **polyfit** in the application launcher |

Alternatively, from the polyfit directory: `python gui.py`

### GUI walkthrough

The GUI walks you through nine steps:

1. **Welcome** — overview of the method and all data formats
2. **Data File** — browse to your measurement file
3. **Uncertainties** — specify errors on $X$, $F$, and optionally $f_0$
4. **Model** — choose Mayo-Lewis, Skeist, or a custom two-parameter model
5. **Prior & Grid** — set the parameter search range and grid resolution
6. **Goodness of Fit** — optionally enable the PTE test
7. **Labels & Colours** — axis labels and plot colours
8. **Output Files** — choose where to save results
9. **Run** — launch the fit and watch the live log

Click **Next / Back** to move between steps.
The GUI remembers your settings between sessions.

---

## Option 2 — Jupyter notebook (pip install)

*For Python users who want interactive analysis or to inspect the statistics in detail.*

### Install

From the polyfit directory:

```bash
pip install -e .
```

This installs polyfit and all its dependencies (`numpy`, `scipy`, `matplotlib`)
in editable mode so any local changes are picked up immediately.
To also run the example notebook:

```bash
pip install jupyter
```

### Run the example notebook

```bash
jupyter notebook examples/example_usage.ipynb
```

The notebook covers:

- Why ordinary least squares is insufficient for copolymerization data
- The EVM covariance framework and effective covariance matrix $R$
- Running a Mayo-Lewis fit (`examples/mayo_lewis_example.ini`)
- Reading the corner plot and credible intervals
- The PTE goodness-of-fit test
- Running a Skeist fit (`examples/skeist_example.ini`, IUPAC reference data)
- Reporting results for publication

---

## Option 3 — Command line with a config file

*For batch processing, scripting, or integrating polyfit into a larger workflow.*

### Install

Same as Option 2 — from the polyfit directory:

```bash
pip install -e .
```

### Run

```bash
python main.py                                 # uses input.ini in the current directory
python main.py examples/mayo_lewis_example.ini # low-conversion example
python main.py examples/skeist_example.ini     # high-conversion example (IUPAC data)
python main.py my_experiment.ini               # your own config file
```

The two ready-to-run example configs in `examples/` show all available options with
inline comments:

- [`examples/mayo_lewis_example.ini`](examples/mayo_lewis_example.ini) — Mayo-Lewis, plain text data, 5 % relative errors
- [`examples/skeist_example.ini`](examples/skeist_example.ini) — Skeist, 4-column CSV, PTE test enabled

---

## Data file formats

polyfit auto-detects the format from the file extension and number of columns.

### Plain text (`.txt`) — 2 columns

```
# independent   dependent
0.10   0.18
0.30   0.40
0.50   0.55
```

Lines starting with `#` are treated as comments.
Column 1 = independent variable $X$ (e.g. feed composition $f_2$),
column 2 = dependent variable $F$ (e.g. copolymer composition $F_2$).
**Uncertainties must be specified in the config file.**

---

### CSV 4-column (`.csv`)

```
f0,X,F,dF
0.10,0.12,0.20,0.01
0.30,0.31,0.38,0.02
```

| Column | Meaning |
|---|---|
| `f0` | Initial feed composition — used as a grouping key for Skeist curves |
| `X` | Measured independent variable |
| `F` | Measured dependent variable |
| `dF` | Absolute $1\sigma$ uncertainty on $F$ |

Uncertainty on $X$ is assumed zero.
Config uncertainty settings are ignored.

---

### CSV 5-column — EVM (`.csv`)

```
f0,X,dX,F,dF
0.10,0.12,0.005,0.20,0.01
0.30,0.31,0.008,0.38,0.02
```

| Column | Meaning |
|---|---|
| `f0` | Initial feed composition |
| `X` | Measured independent variable |
| `dX` | Absolute $1\sigma$ uncertainty on $X$ |
| `F` | Measured dependent variable |
| `dF` | Absolute $1\sigma$ uncertainty on $F$ |

Uncertainties on **both** $X$ and $F$ are propagated via the effective covariance matrix:

$$R = C_{YY} - C_{XY}T - TC_{XY}^\top + TC_{XX}T, \qquad T_{ii} = \frac{\partial F}{\partial X}\bigg|_{x_i}$$

---

### CSV 6-column — EVM + $f_0$ uncertainty (`.csv`)

```
f0,X,dX,F,dF,df0
0.05,0.093,0.005,0.381,0.030,0.002
```

| Column | Meaning |
|---|---|
| `f0` | Initial feed mole fraction of M2 |
| `X` | Fractional conversion $X_\text{conv}$ |
| `dX` | Absolute $1\sigma$ uncertainty on $X_\text{conv}$ |
| `F` | Cumulative copolymer composition $F_\text{cum}$ |
| `dF` | Absolute $1\sigma$ uncertainty on $F_\text{cum}$ |
| `df0` | Absolute $1\sigma$ uncertainty on $f_0$ |

**Skeist model only.** The $f_0$ uncertainty is propagated as an additional term:

$$R \mathrel{+}= T_{f_0} \cdot C_{FF} \cdot T_{f_0}, \qquad T_{f_0,i} = \frac{\partial F_\text{cum}}{\partial f_0}\bigg|_{x_i}$$

---

## Understanding the outputs

### Output files

| File | Content |
|---|---|
| `contour.pdf` | Corner plot: 2D posterior contours at 68.3 / 95.4 / 99.7 % + 1D marginals |
| `bestfit.pdf` | Data with the best-fit model overlaid (one curve per $f_0$ for Skeist) |
| `results.txt` | Best-fit parameters, symmetric Gaussian $\pm\sigma$, and true asymmetric 68 % credible intervals |
| `posterior.txt` | Full 2D posterior grid as a plain-text table |
| `bestfit_curve.txt` | Tabulated best-fit curve values |
| `pte_test.pdf` | PTE goodness-of-fit histogram (only written when `run_pte = true`) |

### Corner plot

The corner plot shows three panels:

- **Bottom-left** — joint 2D posterior with exact credible contours (coloured) and the Gaussian (Fisher matrix) ellipse (dashed).
  If the ellipse matches the contours well the Gaussian approximation is adequate; if not, use the true credible intervals.
- **Top** — 1D marginal posterior for $p_1$ (exact = solid, Gaussian = dashed).
- **Right** — 1D marginal posterior for $p_2$.

### Credible intervals

`results.txt` reports two kinds of $1\sigma$ interval for each parameter:

- **Gaussian** $\pm\sigma$ — symmetric, from the inverse Fisher matrix. Fast but approximate.
- **True 68 %** $-\sigma^- / +\sigma^+$ — asymmetric, from the marginal posterior CDF. Exact.

When the posterior is symmetric the two agree.
When it is skewed (common near physical boundaries) only the true interval is correct.

### PTE goodness-of-fit test

The PTE (Probability To Exceed) answers: *if the best-fit model were the true model,
how often would a new experiment give a worse $\chi^2$ than observed?*

| PTE | Interpretation |
|---|---|
| < 0.05 | Poor fit — the model is likely misspecified or errors are under-estimated |
| 0.05 – 0.95 | Acceptable fit |
| > 0.95 | Suspiciously good — errors may be over-estimated |

The test runs $N_\text{mock}$ re-fits from synthetic data and plots the resulting
$\chi^2$ distribution alongside the observed value.

---

## Full configuration reference

All keys and their defaults.

```ini
[data_structure]
file_data       = data/data.txt   # path to the data file
high_conversion = false           # false = Mayo-Lewis,  true = Skeist

# Uncertainties for plain-text files (ignored for CSV files).
# Format:  rel 0.05  |  abs 0.02  |  path/to/matrix.txt
X_covariance    = rel 0.05        # uncertainty on X
Y_covariance    = rel 0.05        # uncertainty on F  (always required)
XY_covariance   = rel 0.0         # X-F cross-covariance (0 if independent)

# f0 uncertainty — plain-text Skeist data only (6-col CSV uses column 6).
# f0_covariance = rel 0.01


[inference]
prior_range = 0.0, 5.0, 0.0, 5.0  # p1_min, p1_max, p2_min, p2_max

# PTE goodness-of-fit test
run_pte = false   # set to true to enable
N_pte   = 200     # number of mock datasets (100–1000 recommended)


[precision]
N_inter = 100     # grid points per parameter axis  (run time ~ N_inter²)


[plotting]
plot_line_colour          = #00689D
plot_data_colour          = red
independent_variable_name = $f_2$
dependent_variable_name   = $F_2$
parameter_1_name          = $r_2$
parameter_2_name          = $r_1$
contour_plot_zoom         = 1.0    # 1.0 = auto (±10σ);  0.5 = 2× zoom
use_tex                   = False  # True requires latex + dvipng on PATH


[output]
file_name_contour_plot  = output/contour.pdf
file_name_best_fit_plot = output/bestfit.pdf
file_name_results       = output/results.txt
file_name_contour       = output/posterior.txt
file_name_best_fit      = output/bestfit_curve.txt
pte_output_file         = output/pte_test.pdf   # only written when run_pte = true
```

---

## Citing polyfit

If you use polyfit in a publication, please cite:

> R. Reischke, *Macromolecular Theory and Simulations* **32** (2023) 2200063.
> DOI: [10.1002/mats.202200063](https://onlinelibrary.wiley.com/doi/10.1002/mats.202200063)
