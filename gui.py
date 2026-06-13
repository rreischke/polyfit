"""
polyfit GUI — guided wizard for running polyfit without editing config files.

Usage:
    python gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, colorchooser
import base64
import configparser
import os
import sys
import threading
import numpy as np

# Always run relative to the project root so ./data/ etc. resolve correctly.
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Force the non-interactive Agg backend before any matplotlib import.
# FitModel creates figures inside a background thread; GUI backends (TkAgg,
# MacOSX, Qt5, …) require figure creation on the main thread and will raise
# a UserWarning or crash. Agg renders to files only, which is all we need.
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Built-in model templates
# ---------------------------------------------------------------------------

MODELS = {
    "Mayo-Lewis  (copolymerization)": {
        "hint": (
            "Terminal model for instantaneous copolymer composition.\n"
            "Valid when fractional conversion X < ~5 %.\n\n"
            "  p1 = r2 — reactivity ratio of monomer M2\n"
            "            r2 < 1: M2 prefers to cross-propagate (insert M1)\n"
            "            r2 > 1: M2 prefers self-propagation (insert M2)\n"
            "  p2 = r1 — reactivity ratio of monomer M1 (same convention)\n\n"
            "Data format:  2-column plain text (X = f₂, F = F₂)\n"
            "              or 4/5-column CSV (f0, X, [dX,] F, dF)"
        ),
        "code": """\
def model(parameters, f2):
    r1, r2 = parameters[1], parameters[0]
    num = r2 * f2**2 + f2 * (1 - f2)
    den = r2 * f2**2 + 2 * f2 * (1 - f2) + r1 * (1 - f2)**2
    return num / den

def dmodel(parameters, f2):
    r1, r2 = parameters[1], parameters[0]
    den  = r2 * f2**2 + 2 * f2 * (1 - f2) + r1 * (1 - f2)**2
    dden = 2 * f2 * r2 + 2 - 4 * f2 - 2 * r1 * (1 - f2)
    return ((2 * r2 * f2 - 2 * f2 + 1) / den
            - (r2 * f2**2 + f2 * (1 - f2)) * dden / den**2)
""",
    },
    "Skeist  (high conversion, X > 5%)": {
        "hint": (
            "Integrated copolymerization model for high fractional conversion.\n"
            "Use this when X > ~5 % and the feed composition drifts during reaction.\n\n"
            "  p1 = r2 — reactivity ratio of monomer M2\n"
            "  p2 = r1 — reactivity ratio of monomer M1\n\n"
            "Required data columns (CSV):\n"
            "  col 1: f0      — initial feed mole fraction of M2\n"
            "  col 2: X_conv  — fractional monomer conversion (0–1)\n"
            "  col 3: dX      — absolute 1σ uncertainty on X_conv  (5-col CSV)\n"
            "  col 4: F_cum   — measured cumulative copolymer composition\n"
            "  col 5: dF_cum  — absolute 1σ uncertainty on F_cum\n"
            "  col 6: df0     — absolute 1σ uncertainty on f0  (6-col CSV only)\n\n"
            "Note: if df0 is provided in a 6-column CSV, f0 uncertainty is\n"
            "automatically propagated through the fit (dmodel_f0 is used).\n\n"
            "This model is slower than Mayo-Lewis — use N_inter ≤ 150."
        ),
        "high_conversion": True,
        "code": """\
from source.models import skeist, skeist_deriv, skeist_deriv_f0

def model(parameters, x):
    \"\"\"Cumulative copolymer composition F_cum(r2, r1 | f0, X_conv).\"\"\"
    return skeist(parameters, x)

def dmodel(parameters, x):
    \"\"\"Analytical dF_cum/dX_conv — for X-error propagation.\"\"\"
    return skeist_deriv(parameters, x)

def dmodel_f0(parameters, x):
    \"\"\"Analytical dF_cum/df0 — for f0-error propagation.
    Used automatically when f0 uncertainties are present (6-col CSV
    or f0_covariance in the config).  Can be removed if not needed.\"\"\"
    return skeist_deriv_f0(parameters, x)
""",
    },
    "Linear  y = p1·x + p2": {
        "hint": (
            "Straight-line model.\n"
            "  p1 = slope\n"
            "  p2 = y-intercept"
        ),
        "code": """\
def model(parameters, x):
    return parameters[0] * x + parameters[1]

def dmodel(parameters, x):
    return parameters[0]
""",
    },
    "Power law  y = p1·x^p2": {
        "hint": (
            "Power-law model.\n"
            "  p1 = amplitude\n"
            "  p2 = exponent"
        ),
        "code": """\
def model(parameters, x):
    return parameters[0] * x ** parameters[1]

def dmodel(parameters, x):
    return parameters[0] * parameters[1] * x ** (parameters[1] - 1)
""",
    },
    "Custom — enter below": {
        "hint": (
            "Define model(parameters, x) and dmodel(parameters, x).\n"
            "  parameters[0] = p1,  parameters[1] = p2\n\n"
            "  dmodel must return  d(model)/dx  (not d/dp).\n"
            "  This is the derivative w.r.t. the independent variable x.\n"
            "  It is used to propagate x-uncertainties into the fit via EVM.\n\n"
            "  Optionally also define dmodel_f0(parameters, x) if your\n"
            "  model depends on a second input with its own uncertainty."
        ),
        "code": """\
def model(parameters, x):
    # Replace with your model
    return parameters[0] * x + parameters[1]

def dmodel(parameters, x):
    # Replace with d(model)/dx
    return parameters[0]
""",
    },
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

BG         = "white"
SIDEBAR_BG = "#1E3A5F"
MUTED      = "#8BAFD0"
DONE_COL   = "#4CAF50"

# File where the GUI saves its state between sessions.
STATE_FILE = os.path.join(os.path.expanduser("~"), ".polyfit_gui_state.ini")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class PolyfitGUI(tk.Tk):

    STEPS = [
        "Welcome",
        "Data File",
        "Uncertainties",
        "Model",
        "Prior & Grid",
        "Goodness of Fit",
        "Labels & Colours",
        "Output Files",
        "Run",
    ]

    def __init__(self):
        super().__init__()
        self.title("polyfit — Guided Setup")
        self.minsize(760, 640)
        self._step = 0
        self._frames: list[tk.Frame] = []
        self._build_chrome()
        self._build_all_steps()
        self._show_step(0)
        self._load_state()   # restore last session — called after all widgets exist

    # ------------------------------------------------------------------ #
    # Chrome: sidebar + content area + nav bar
    # ------------------------------------------------------------------ #

    def _build_chrome(self):
        top = tk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True)

        # sidebar
        self._sb = tk.Frame(top, bg=SIDEBAR_BG, width=190)
        self._sb.pack(side=tk.LEFT, fill=tk.Y)
        self._sb.pack_propagate(False)
        tk.Label(self._sb, text="polyfit", font=("Arial", 17, "bold"),
                 bg=SIDEBAR_BG, fg="white", pady=20).pack(fill=tk.X)
        self._sb_labels: list[tk.Label] = []
        for i, name in enumerate(self.STEPS):
            lbl = tk.Label(self._sb, text=f"  {i + 1}. {name}",
                           font=("Arial", 10), anchor="w",
                           bg=SIDEBAR_BG, fg=MUTED, padx=12, pady=6)
            lbl.pack(fill=tk.X)
            self._sb_labels.append(lbl)

        # citation at the bottom of the sidebar
        tk.Label(self._sb,
                 text="Reischke 2023\nMacromol. Theory Simul.\nDOI: 10.1002/mats.202200063",
                 font=("Arial", 7), bg=SIDEBAR_BG, fg="#567A9E",
                 justify=tk.CENTER, wraplength=160, pady=8,
                 ).pack(side=tk.BOTTOM, fill=tk.X)

        # content area
        self._content = tk.Frame(top, bg=BG)
        self._content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # nav bar
        nav = tk.Frame(self, relief=tk.GROOVE, bd=1)
        nav.pack(side=tk.BOTTOM, fill=tk.X)
        self._btn_back = ttk.Button(nav, text="← Back", command=self._go_back)
        self._btn_back.pack(side=tk.LEFT, padx=10, pady=8)
        self._btn_next = ttk.Button(nav, text="Next →", command=self._go_next)
        self._btn_next.pack(side=tk.RIGHT, padx=10, pady=8)

    # ------------------------------------------------------------------ #
    # Generic layout helpers
    # ------------------------------------------------------------------ #

    def _make_step_frame(self, title: str, subtitle: str = "") -> tk.Frame:
        f = tk.Frame(self._content, bg=BG)
        tk.Label(f, text=title, font=("Arial", 18, "bold"),
                 bg=BG, fg=SIDEBAR_BG).pack(anchor="w", padx=30, pady=(26, 0))
        if subtitle:
            tk.Label(f, text=subtitle, font=("Arial", 10),
                     bg=BG, fg="#555").pack(anchor="w", padx=30, pady=(2, 6))
        ttk.Separator(f).pack(fill="x", padx=30, pady=(4, 14))
        return f

    @staticmethod
    def _field(parent, label: str, var, width: int = None):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill=tk.X, padx=30, pady=4)
        tk.Label(row, text=label, font=("Arial", 10), bg=BG,
                 width=26, anchor="w").pack(side=tk.LEFT)
        kw = {"textvariable": var}
        if width:
            kw["width"] = width
        ttk.Entry(row, **kw).pack(side=tk.LEFT, fill=tk.X, expand=True)

    @staticmethod
    def _browse_field(parent, label: str, var, filetypes=(("All files", "*.*"),)):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill=tk.X, padx=30, pady=4)
        tk.Label(row, text=label, font=("Arial", 10), bg=BG,
                 width=26, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse…",
                   command=lambda: var.set(
                       filedialog.askopenfilename(filetypes=list(filetypes)) or var.get()
                   )).pack(side=tk.LEFT, padx=(6, 0))

    @staticmethod
    def _saveas_field(parent, label: str, var):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill=tk.X, padx=30, pady=3)
        tk.Label(row, text=label, font=("Arial", 10), bg=BG,
                 width=26, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse…",
                   command=lambda: var.set(
                       filedialog.asksaveasfilename(
                           defaultextension=os.path.splitext(var.get())[1] or ".pdf",
                           filetypes=[("PDF", "*.pdf"), ("PNG", "*.png"),
                                      ("Text", "*.txt"), ("All files", "*.*")]
                       ) or var.get()
                   )).pack(side=tk.LEFT, padx=(6, 0))

    def _colour_row(self, parent, label: str, var):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill=tk.X, padx=30, pady=4)
        tk.Label(row, text=label, font=("Arial", 10), bg=BG,
                 width=26, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT)
        swatch = tk.Label(row, width=3, relief="solid")
        swatch.pack(side=tk.LEFT, padx=6)
        self._sync_swatch(var, swatch)
        var.trace_add("write", lambda *_: self._sync_swatch(var, swatch))
        ttk.Button(row, text="Pick…",
                   command=lambda v=var, s=swatch: self._pick_colour(v, s)
                   ).pack(side=tk.LEFT)

    @staticmethod
    def _sync_swatch(var, swatch):
        try:
            swatch.config(bg=var.get())
        except Exception:
            pass

    @staticmethod
    def _pick_colour(var, swatch):
        colour = colorchooser.askcolor(color=var.get())[1]
        if colour:
            var.set(colour)
            try:
                swatch.config(bg=colour)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Covariance block helper (used in Step 2)
    # ------------------------------------------------------------------ #

    def _cov_block(self, parent, label: str,
                   mode_var, value_var, file_var):
        box = tk.LabelFrame(parent, text=label, bg=BG,
                            font=("Arial", 10, "bold"), padx=10, pady=6)
        box.pack(fill=tk.X, padx=4, pady=6)

        for text, val in [
            ("Relative error  (e.g. 0.05 = 5 %)", "rel"),
            ("Absolute error  (constant value)",   "abs"),
            ("External matrix file (.txt)",         "file"),
        ]:
            ttk.Radiobutton(box, text=text,
                            variable=mode_var, value=val).pack(anchor="w")

        val_row = tk.Frame(box, bg=BG)
        val_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(val_row, text="Value:", bg=BG, width=8,
                 anchor="w").pack(side=tk.LEFT)
        val_entry = ttk.Entry(val_row, textvariable=value_var, width=10)
        val_entry.pack(side=tk.LEFT)

        file_row = tk.Frame(box, bg=BG)
        file_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(file_row, text="File:", bg=BG, width=8,
                 anchor="w").pack(side=tk.LEFT)
        file_entry = ttk.Entry(file_row, textvariable=file_var)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_row, text="Browse…",
                   command=lambda: file_var.set(
                       filedialog.askopenfilename(
                           filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                       ) or file_var.get()
                   )).pack(side=tk.LEFT, padx=(6, 0))

        def _sync_state(*_):
            m = mode_var.get()
            val_entry.config(state="normal"   if m in ("rel", "abs") else "disabled")
            file_entry.config(state="normal"  if m == "file"         else "disabled")

        mode_var.trace_add("write", _sync_state)
        _sync_state()

    # ------------------------------------------------------------------ #
    # Build every step frame
    # ------------------------------------------------------------------ #

    def _build_all_steps(self):
        self._s_welcome()
        self._s_data()
        self._s_covariance()
        self._s_model()
        self._s_prior()
        self._s_pte()
        self._s_plotting()
        self._s_output()
        self._s_run()

    # ── Step 0: Welcome ──────────────────────────────────────────────── #

    def _s_welcome(self):
        f = self._make_step_frame("Welcome to polyfit")

        canvas = tk.Canvas(f, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        def section(title, body):
            tk.Label(inner, text=title, font=("Arial", 11, "bold"),
                     bg=BG, fg=SIDEBAR_BG, justify=tk.LEFT
                     ).pack(anchor="w", padx=30, pady=(14, 2))
            tk.Label(inner, text=body, font=("Courier", 9),
                     bg=BG, fg="#333", justify=tk.LEFT
                     ).pack(anchor="w", padx=30)

        section("What does polyfit do?",
            "polyfit fits a two-parameter non-linear model to measured\n"
            "data and returns a best-fit parameter vector together with\n"
            "proper Bayesian credible regions.\n"
            "Unlike a simple least-squares fit, it:\n"
            "  •  propagates errors on BOTH x and y (Errors-in-Variables)\n"
            "  •  handles correlated errors between x and y\n"
            "  •  evaluates the full 2-D posterior on a grid — credible\n"
            "     regions are exact, not just Gaussian ellipses\n"
            "  •  provides true asymmetric 1-D credible intervals from\n"
            "     the marginal posteriors")

        section("What are credible regions?",
            "A 68.3 % credible region is the set of parameter values\n"
            "(r1, r2) inside which the posterior integrates to 68.3 %.\n"
            "It is the Bayesian analogue of a 1-sigma confidence region.\n"
            "Because the posterior can be skewed or banana-shaped, the\n"
            "true region is often not an ellipse.  polyfit shows both:\n"
            "the exact contour and the Gaussian (Fisher matrix) ellipse,\n"
            "so you can judge whether the Gaussian approximation holds.")

        section("What is the Errors-in-Variables (EVM) approach?",
            "Standard least squares assumes only y carries errors.\n"
            "EVM propagates errors from ALL variables into an effective\n"
            "covariance matrix R:\n"
            "\n"
            "   R = C_YY - C_XY·T - T·C_XY + T·C_XX·T\n"
            "\n"
            "T = diag(dmodel/dx) evaluated at each data point.\n"
            "C_XX, C_YY, C_XY are the x-, y-, and cross-covariance matrices.\n"
            "\n"
            "If x and y are independent (C_XY = 0) this simplifies to:\n"
            "   R = C_YY + (dF/dx)^2 · C_XX\n"
            "\n"
            "A large dF/dx amplifies x-errors; polyfit accounts for this\n"
            "automatically via the derivative T.")

        section("What is the Skeist model (high conversion)?",
            "At low conversion (X < ~5 %) the feed composition barely\n"
            "changes, so the Mayo-Lewis instantaneous equation is a good\n"
            "approximation.  At higher conversion the feed drifts because\n"
            "the more reactive monomer is consumed faster.  The Skeist\n"
            "model integrates Mayo-Lewis over the full conversion path.\n"
            "\n"
            "The independent variable becomes X_conv (fractional conversion)\n"
            "and each experiment starts from an initial feed f0, so both\n"
            "f0 and X_conv appear in the data.  If f0 also has a measurement\n"
            "uncertainty it can be propagated via dF_cum/df0 (6-col CSV).")

        section("Supported data formats",
            "Plain text  (.txt)  2 col:  X, F\n"
            "                    Uncertainties set manually in Step 3.\n"
            "\n"
            "CSV 4-col   (.csv)  f0, X, F, dF\n"
            "                    dF = 1-sigma uncertainty on F.\n"
            "                    X assumed error-free.\n"
            "\n"
            "CSV 5-col   (.csv)  f0, X, dX, F, dF\n"
            "                    Both X and F errors via EVM.\n"
            "\n"
            "CSV 6-col   (.csv)  f0, X, dX, F, dF, df0\n"
            "                    Also propagates f0 uncertainty (Skeist).\n"
            "\n"
            "For CSV files the error columns take priority — manual\n"
            "uncertainty settings in Step 3 are ignored.")

        section("What outputs do you get?",
            "  corner_plot.pdf     2-D posterior contours at 68/95/99 %\n"
            "                      + 1-D marginals (combined corner layout)\n"
            "  bestfit_plot.pdf    Data + best-fit curve overlay\n"
            "                      (one curve per f0 for Skeist)\n"
            "  results.txt         Best-fit, Gaussian +/-sigma, and true\n"
            "                      asymmetric 68 % credible intervals\n"
            "  posterior.txt       Raw posterior grid as a text table\n"
            "  bestfit_curve.txt   Tabulated best-fit curve values\n"
            "  pte_test.pdf        PTE goodness-of-fit histogram (optional)")

        tk.Label(inner,
                 text="\nClick  Next →  to begin.\n\n"
                      "─────────────────────────────────────────────────────\n"
                      "Reference:  R. Reischke, Macromol. Theory Simul. 32 (2023) 2200063\n"
                      "DOI: 10.1002/mats.202200063",
                 font=("Arial", 9), bg=BG, fg="#888",
                 justify=tk.LEFT).pack(anchor="w", padx=30, pady=(16, 4))

        self._frames.append(f)

    # ── Step 1: Data file ─────────────────────────────────────────────── #

    def _s_data(self):
        f = self._make_step_frame(
            "Data File",
            "Select the file containing your measurements.")

        self._v_data = tk.StringVar(value="./data/data.txt")
        self._browse_field(f, "Data file:", self._v_data,
                           filetypes=[("Text / CSV", "*.txt *.csv"),
                                      ("All files", "*.*")])

        fmt_box = tk.LabelFrame(f, text="Supported column formats", bg=BG, padx=10, pady=6)
        fmt_box.pack(fill=tk.X, padx=30, pady=(10, 4))
        tk.Label(fmt_box, text=(
            "Plain text (.txt) — 2 columns, whitespace-separated, no header\n"
            "                    col 1: X    independent variable (e.g. f₂, feed composition)\n"
            "                    col 2: F    dependent variable   (e.g. F₂, copolymer composition)\n"
            "                    → set uncertainties manually in Step 3\n\n"
            "CSV 4-col (.csv)  — comma-separated, header optional\n"
            "                    col 1: f0   nominal feed composition (reference / Skeist grouping)\n"
            "                    col 2: X    measured independent variable\n"
            "                    col 3: F    measured dependent variable\n"
            "                    col 4: dF   absolute 1σ uncertainty on F\n"
            "                    → X is assumed error-free\n\n"
            "CSV 5-col (.csv)  — as above, plus:\n"
            "                    col 3: dX   absolute 1σ uncertainty on X  (EVM)\n"
            "                    col 4: F,  col 5: dF\n"
            "                    → both X and F errors propagated via EVM\n\n"
            "CSV 6-col (.csv)  — as 5-col, plus:\n"
            "                    col 6: df0  absolute 1σ uncertainty on f0\n"
            "                    → f0 error propagated (Skeist model only)"
        ), font=("Courier", 9), bg=BG, fg="#333", justify=tk.LEFT).pack(anchor="w")

        pf = tk.LabelFrame(f, text="Preview  (first 5 rows)", bg=BG, padx=6, pady=4)
        pf.pack(fill=tk.X, padx=30, pady=(8, 0))
        self._preview_text = tk.Text(pf, height=5, font=("Courier", 10),
                                     state=tk.DISABLED, bg="#F4F4F4")
        self._preview_text.pack(fill=tk.X)

        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(fill=tk.X, padx=30, pady=6)
        ttk.Button(btn_row, text="Load preview", command=self._do_preview).pack(side=tk.LEFT)
        self._fmt_label = tk.Label(btn_row, text="", font=("Arial", 9),
                                   bg=BG, fg="#007700")
        self._fmt_label.pack(side=tk.LEFT, padx=12)
        self._frames.append(f)

    @staticmethod
    def _detect_data_format(path):
        """Return a human-readable format string for display in the GUI."""
        ext = os.path.splitext(path)[1].lower()
        with open(path, 'r') as fh:
            first = fh.readline().strip()
        is_csv = (ext == '.csv') or (',' in first)
        if not is_csv:
            return "Plain text  (2-column: X, F)"
        try:
            float(first.split(',')[0].strip())
            skiprows = 0
        except ValueError:
            skiprows = 1
        data = np.loadtxt(path, delimiter=',', skiprows=skiprows)
        n = data.shape[1] if data.ndim > 1 else len(data)
        if n == 4:
            return "CSV 4-col  (f0, X, F, dF) — errors from file"
        if n == 5:
            return "CSV 5-col EVM  (f0, X, dX, F, dF) — errors from file"
        if n == 6:
            return "CSV 6-col EVM+f0  (f0, X, dX, F, dF, df0) — errors from file"
        return f"CSV — unrecognised ({n} columns)"

    def _do_preview(self):
        self._preview_text.config(state=tk.NORMAL)
        self._preview_text.delete("1.0", tk.END)
        path = self._v_data.get()
        try:
            with open(path) as fh:
                self._preview_text.insert(
                    tk.END, "".join(fh.readline() for _ in range(5)))
            fmt = self._detect_data_format(path)
            self._fmt_label.config(text=f"Detected: {fmt}", fg="#007700")
        except Exception as exc:
            self._preview_text.insert(tk.END, f"Cannot read file: {exc}")
            self._fmt_label.config(text="Could not detect format", fg="#CC0000")
        self._preview_text.config(state=tk.DISABLED)

    # ── Step 2: Uncertainties ─────────────────────────────────────────── #

    def _s_covariance(self):
        f = self._make_step_frame(
            "Measurement Uncertainties",
            "Define how errors in x, y, and f0 (Skeist only) enter the fit.")

        canvas = tk.Canvas(f, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        note = tk.LabelFrame(inner, text="When do these settings apply?",
                             bg="#FFF8E7", padx=10, pady=6)
        note.pack(fill=tk.X, padx=4, pady=(0, 10))
        tk.Label(note, text=(
            "CSV files (4/5/6 columns):  errors are read from the dX, dF, df0 columns.\n"
            "These settings below are IGNORED for CSV files.\n\n"
            "Plain text files (2 columns):  errors MUST be set here.\n"
            "  Choose 'Relative error' for a percentage uncertainty (e.g. 0.05 = 5 %)\n"
            "  or 'Absolute error' for a fixed ±value.\n"
            "  For a full N×N covariance matrix (correlated errors), supply a text file."
        ), font=("Arial", 9), bg="#FFF8E7", fg="#555", justify=tk.LEFT).pack(anchor="w")

        expl = tk.LabelFrame(inner, text="What do the three covariance terms mean?",
                             bg="#F0F4FF", padx=10, pady=6)
        expl.pack(fill=tk.X, padx=4, pady=(0, 6))
        tk.Label(expl, text=(
            "C_XX  — variance (squared uncertainty) in the independent variable x.\n"
            "         Example: x = feed composition f₂ prepared gravimetrically.\n"
            "         If f₂ is very precisely prepared, set X_covariance = rel 0.0.\n"
            "         If weighing introduces ~1 % error, use rel 0.01.\n\n"
            "C_YY  — variance in the dependent variable y (always required for plain text).\n"
            "         Example: y = NMR-measured copolymer composition F₂.\n"
            "         Typical NMR precision: 2–5 % relative → rel 0.02 to 0.05.\n\n"
            "C_XY  — cross-covariance between x and y errors.\n"
            "         Use rel 0.0 if x and y are measured independently.\n"
            "         Use a non-zero value only if BOTH x and y come from the SAME\n"
            "         measurement (e.g. two integrals from a single NMR spectrum;\n"
            "         in that case errors in the spectrum shift both f₂ and F₂\n"
            "         simultaneously, creating a positive correlation)."
        ), font=("Arial", 9), bg="#F0F4FF", fg="#333", justify=tk.LEFT).pack(anchor="w")

        self._cov_xm  = tk.StringVar(value="rel")
        self._cov_xv  = tk.StringVar(value="0.0")
        self._cov_xf  = tk.StringVar()
        self._cov_ym  = tk.StringVar(value="rel")
        self._cov_yv  = tk.StringVar(value="0.05")
        self._cov_yf  = tk.StringVar()
        self._cov_xym = tk.StringVar(value="rel")
        self._cov_xyv = tk.StringVar(value="0.0")
        self._cov_xyf = tk.StringVar()

        self._cov_block(inner, "X covariance  C_XX  (independent variable, e.g. f₂)",
                        self._cov_xm, self._cov_xv, self._cov_xf)
        self._cov_block(inner, "Y covariance  C_YY  (dependent variable, e.g. F₂) — required",
                        self._cov_ym, self._cov_yv, self._cov_yf)
        self._cov_block(inner, "XY cross-covariance  C_XY  (correlation between x and y errors)",
                        self._cov_xym, self._cov_xyv, self._cov_xyf)

        f0_expl = tk.LabelFrame(inner,
                                text="f0 covariance  C_FF  (Skeist high-conversion model only)",
                                bg="#F4FFF4", padx=10, pady=6)
        f0_expl.pack(fill=tk.X, padx=4, pady=6)
        tk.Label(f0_expl, text=(
            "For the Skeist model, each experiment also has an initial feed composition f0.\n"
            "If f0 carries measurement uncertainty (e.g. from the mass of monomer weighed in),\n"
            "that error can be propagated through the model via the derivative dF_cum/df0.\n\n"
            "For 6-column CSV files:  df0 is read from column 6 — leave the field below empty.\n"
            "For plain-text Skeist data:  enter the f0 uncertainty here.\n\n"
            "For all other models (Mayo-Lewis, custom):  this field is ignored."
        ), font=("Arial", 9), bg="#F4FFF4", fg="#333", justify=tk.LEFT).pack(anchor="w")

        self._cov_ffm = tk.StringVar(value="rel")
        self._cov_ffv = tk.StringVar(value="0.0")
        self._cov_fff = tk.StringVar()
        self._cov_block(inner, "f0 covariance  C_FF  (plain-text Skeist only)",
                        self._cov_ffm, self._cov_ffv, self._cov_fff)

        self._frames.append(f)

    # ── Step 3: Model ─────────────────────────────────────────────────── #

    def _s_model(self):
        f = self._make_step_frame(
            "Model",
            "Choose a built-in model or write your own.")

        intro = tk.LabelFrame(f, text="What is the model used for?", bg="#F0F4FF",
                              padx=10, pady=6)
        intro.pack(fill=tk.X, padx=30, pady=(0, 8))
        tk.Label(intro, text=(
            "The model F = model(parameters, x) maps the independent variable x and\n"
            "the two free parameters (p1, p2) to a predicted value F.\n\n"
            "dmodel(parameters, x) is its derivative w.r.t. x.  It is needed to\n"
            "compute the EVM effective covariance  R = C_YY + T²·C_XX  (see Step 3).\n"
            "If dmodel returns zero everywhere (e.g. no x-errors), the EVM reduces\n"
            "to a standard chi-squared fit with only y-errors.\n\n"
            "Optionally, dmodel_f0(parameters, x) can be defined to also propagate\n"
            "uncertainty in f0 (the initial feed composition, Skeist model only).\n"
            "The Skeist template below already includes all three functions."
        ), font=("Arial", 9), bg="#F0F4FF", fg="#333", justify=tk.LEFT).pack(anchor="w")

        self._v_model = tk.StringVar(value=list(MODELS)[0])

        row = tk.Frame(f, bg=BG)
        row.pack(fill=tk.X, padx=30, pady=4)
        tk.Label(row, text="Model:", font=("Arial", 10), bg=BG,
                 width=10, anchor="w").pack(side=tk.LEFT)
        cb = ttk.Combobox(row, textvariable=self._v_model,
                          values=list(MODELS), state="readonly", width=46)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", self._load_model_template)

        self._model_hint = tk.Label(f, text="", font=("Arial", 9),
                                    bg=BG, fg="#555", justify=tk.LEFT)
        self._model_hint.pack(anchor="w", padx=30, pady=(4, 8))

        cf = tk.LabelFrame(f, text="Code  (editable — changes are applied at run time)",
                           bg=BG, padx=6, pady=4)
        cf.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 8))
        self._model_code = scrolledtext.ScrolledText(
            cf, font=("Courier", 10), height=12)
        self._model_code.pack(fill=tk.BOTH, expand=True)

        self._load_model_template(None)
        self._frames.append(f)

    def _load_model_template(self, _event):
        info = MODELS[self._v_model.get()]
        self._model_hint.config(text=info["hint"])
        self._model_code.delete("1.0", tk.END)
        self._model_code.insert(tk.END, info["code"])

    # ── Step 4: Prior & Grid ──────────────────────────────────────────── #

    def _s_prior(self):
        f = self._make_step_frame(
            "Prior Range & Grid Resolution",
            "Set the parameter search space and posterior grid density.")

        canvas = tk.Canvas(f, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        expl = tk.LabelFrame(inner, text="What is the prior range?",
                             bg="#F0F4FF", padx=10, pady=6)
        expl.pack(fill=tk.X, padx=4, pady=(0, 8))
        tk.Label(expl, text=(
            "polyfit uses a flat (uniform) prior: all parameter values within [min, max]\n"
            "are equally probable a priori.  Values outside the range are excluded.\n\n"
            "The prior range is used in two ways:\n"
            "  1.  It is the search space for the initial Nelder-Mead optimisation that\n"
            "      finds the best-fit (MLE) parameters.\n"
            "  2.  After the MLE is found, the grid is automatically tightened to\n"
            "      ±10 Gaussian standard deviations around the best-fit — so the\n"
            "      prior range mainly sets the boundaries for the optimiser.\n\n"
            "Rule of thumb: set the range wide enough to contain the true values,\n"
            "but not so wide that the optimiser gets lost in flat regions.\n"
            "For reactivity ratios, [0, 2] or [0, 5] is usually sufficient.\n"
            "Negative values are physical only if the model permits them."
        ), font=("Arial", 9), bg="#F0F4FF", fg="#333", justify=tk.LEFT).pack(anchor="w")

        self._v_p1min  = tk.StringVar(value="0.0")
        self._v_p1max  = tk.StringVar(value="2.0")
        self._v_p2min  = tk.StringVar(value="0.0")
        self._v_p2max  = tk.StringVar(value="2.0")
        self._v_ninter = tk.StringVar(value="100")

        prior_box = tk.LabelFrame(inner, text="Prior ranges", bg=BG, padx=10, pady=6)
        prior_box.pack(fill=tk.X, padx=4, pady=4)
        for label, vmin, vmax in [
            ("Parameter 1 (p1) range:", self._v_p1min, self._v_p1max),
            ("Parameter 2 (p2) range:", self._v_p2min, self._v_p2max),
        ]:
            row = tk.Frame(prior_box, bg=BG)
            row.pack(fill=tk.X, pady=5)
            tk.Label(row, text=label, font=("Arial", 10), bg=BG,
                     width=26, anchor="w").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=vmin, width=10).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(row, text="to", bg=BG).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=vmax, width=10).pack(side=tk.LEFT, padx=(4, 0))

        grid_box = tk.LabelFrame(inner, text="Grid resolution", bg=BG, padx=10, pady=6)
        grid_box.pack(fill=tk.X, padx=4, pady=4)
        tk.Label(grid_box, text=(
            "The posterior is evaluated on an N_inter × N_inter grid of parameter values.\n"
            "Finer grids give smoother contours and more accurate credible intervals,\n"
            "but the run time scales as N_inter².\n\n"
            "  N_inter = 50   →  fast (~5 s Mayo-Lewis),  rough contours\n"
            "  N_inter = 100  →  good balance for most fits\n"
            "  N_inter = 200  →  high quality, ~4× slower than 100\n"
            "  N_inter = 500  →  publication quality, very slow for Skeist\n\n"
            "For the Skeist model (high conversion) each likelihood evaluation\n"
            "involves numerical integration, so keep N_inter ≤ 150."
        ), font=("Arial", 9), bg=BG, fg="#333", justify=tk.LEFT).pack(anchor="w", pady=(0, 6))

        row = tk.Frame(grid_box, bg=BG)
        row.pack(fill=tk.X)
        tk.Label(row, text="N_inter:", font=("Arial", 10), bg=BG,
                 width=12, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._v_ninter, width=7).pack(side=tk.LEFT)
        tk.Label(row, text="  grid points per axis",
                 font=("Arial", 9), bg=BG, fg="#777").pack(side=tk.LEFT)

        self._frames.append(f)

    # ── Step 5: Goodness of fit (PTE) ─────────────────────────────────── #

    def _s_pte(self):
        f = self._make_step_frame(
            "Goodness-of-Fit Test  (PTE)",
            "Optionally test whether the best-fit model is a statistically acceptable description of the data.")

        canvas = tk.Canvas(f, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        expl = tk.LabelFrame(inner, text="What is the PTE test?",
                             bg="#F0F4FF", padx=10, pady=6)
        expl.pack(fill=tk.X, padx=4, pady=(0, 8))
        tk.Label(expl, text=(
            "PTE = Probability-To-Exceed.  It answers the question:\n\n"
            "  'If the best-fit model were the true model, how often would\n"
            "   a new experiment give a worse chi-squared than the one observed?'\n\n"
            "How it works:\n"
            "  1.  N_mock synthetic datasets are drawn from a multivariate Gaussian\n"
            "      centred on the best-fit model with covariance matrix R(θ̂).\n"
            "  2.  Each mock dataset is re-fitted to obtain its own minimum chi-squared.\n"
            "  3.  PTE = fraction of mocks with chi²_mock ≥ chi²_data.\n\n"
            "Interpretation:\n"
            "  PTE ≈ 0.5  — typical; the model is a good description of the data.\n"
            "  PTE > 0.95 — the data fit suspiciously well (possible over-fitting or\n"
            "               over-estimated uncertainties).\n"
            "  PTE < 0.05 — the fit is poor; the model is likely misspecified or\n"
            "               the error budget is too small.\n\n"
            "The test produces a histogram of chi²_mock values with the observed\n"
            "chi²_data marked as a dotted vertical line.  More mocks → smoother\n"
            "histogram but longer run time (each mock requires a full re-fit)."
        ), font=("Arial", 9), bg="#F0F4FF", fg="#333", justify=tk.LEFT).pack(anchor="w")

        self._v_run_pte = tk.BooleanVar(value=False)
        self._v_n_pte   = tk.StringVar(value="200")

        pte_box = tk.LabelFrame(inner, text="PTE test settings", bg=BG, padx=10, pady=8)
        pte_box.pack(fill=tk.X, padx=4, pady=4)

        chk_row = tk.Frame(pte_box, bg=BG)
        chk_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(chk_row,
                        text="Run PTE goodness-of-fit test after the main fit",
                        variable=self._v_run_pte).pack(side=tk.LEFT)

        n_row = tk.Frame(pte_box, bg=BG)
        n_row.pack(fill=tk.X)
        tk.Label(n_row, text="Number of mock datasets:", font=("Arial", 10),
                 bg=BG, width=26, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(n_row, textvariable=self._v_n_pte, width=7).pack(side=tk.LEFT)
        tk.Label(n_row, text="  N_mock  (100–1000 recommended)",
                 font=("Arial", 9), bg=BG, fg="#777").pack(side=tk.LEFT)

        tk.Label(pte_box,
                 text="\nNote: each mock fit runs Nelder-Mead from the best-fit parameters.\n"
                      "200 mocks takes roughly the same time as the main fit itself.",
                 font=("Arial", 9), bg=BG, fg="#666").pack(anchor="w")

        self._frames.append(f)

    # ── Step 6: Labels & Colours ──────────────────────────────────────── #

    def _s_plotting(self):
        f = self._make_step_frame(
            "Labels & Colours",
            "Customise axis labels and plot colours.  LaTeX notation ($…$) is supported.")

        note = tk.LabelFrame(f, text="LaTeX in labels", bg="#FFF8E7", padx=10, pady=4)
        note.pack(fill=tk.X, padx=30, pady=(0, 8))
        tk.Label(note, text=(
            "Labels support Matplotlib MathText notation: enclose maths in $…$.\n"
            "Example:  $r_2$  renders as r₂,   $F_\\mathrm{cum}$  renders as F_cum.\n"
            "Full LaTeX rendering (requires LaTeX + dvipng installed) can be\n"
            "enabled with the checkbox below.  MathText works without any TeX install."
        ), font=("Arial", 9), bg="#FFF8E7", fg="#555", justify=tk.LEFT).pack(anchor="w")

        self._v_xlabel = tk.StringVar(value="$f_2$")
        self._v_ylabel = tk.StringVar(value="$F_2$")
        self._v_p1lbl  = tk.StringVar(value="$r_2$")
        self._v_p2lbl  = tk.StringVar(value="$r_1$")
        self._v_lcol   = tk.StringVar(value="#00689D")
        self._v_dcol   = tk.StringVar(value="red")
        self._v_tex    = tk.BooleanVar(value=False)
        self._v_zoom   = tk.StringVar(value="1.0")

        self._field(f, "Independent variable (x):", self._v_xlabel)
        self._field(f, "Dependent variable (y):",   self._v_ylabel)
        self._field(f, "Parameter 1 label (p1):",   self._v_p1lbl)
        self._field(f, "Parameter 2 label (p2):",   self._v_p2lbl)
        ttk.Separator(f).pack(fill="x", padx=30, pady=10)

        self._colour_row(f, "Fit line / contour colour:", self._v_lcol)
        self._colour_row(f, "Data point colour:",         self._v_dcol)
        ttk.Separator(f).pack(fill="x", padx=30, pady=10)

        row = tk.Frame(f, bg=BG)
        row.pack(fill=tk.X, padx=30, pady=4)
        ttk.Checkbutton(
            row,
            text="Use LaTeX rendering  (requires a working LaTeX installation)",
            variable=self._v_tex,
        ).pack(side=tk.LEFT)

        zoom_box = tk.Frame(f, bg=BG)
        zoom_box.pack(fill=tk.X, padx=30, pady=4)
        tk.Label(zoom_box, text="Contour plot zoom:", font=("Arial", 10),
                 bg=BG, width=26, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(zoom_box, textvariable=self._v_zoom, width=6).pack(side=tk.LEFT)
        tk.Label(zoom_box,
                 text="  1.0 = auto-range  (±10σ);  0.5 = zoomed in  2×",
                 font=("Arial", 9), bg=BG, fg="#777").pack(side=tk.LEFT)

        self._frames.append(f)

    # ── Step 7: Output files ──────────────────────────────────────────── #

    def _s_output(self):
        f = self._make_step_frame(
            "Output Files",
            "Choose where to save plots and result tables.")

        note = tk.LabelFrame(f, text="What each file contains", bg="#FFF8E7",
                             padx=10, pady=6)
        note.pack(fill=tk.X, padx=30, pady=(0, 8))
        tk.Label(note, text=(
            "Corner plot:       2-D posterior contours at 68.3/95.4/99.7 % +\n"
            "                   1-D marginals in top/right panels (one PDF).\n"
            "Best-fit plot:     Data points with the best-fit model overlaid.\n"
            "                   For Skeist: one curve per unique f0 value.\n"
            "Results summary:   Best-fit values, Gaussian ±σ, and true asymmetric\n"
            "                   68 % credible intervals from the marginal CDFs.\n"
            "Posterior grid:    Raw N_inter × N_inter grid of normalised posterior\n"
            "                   values as a plain-text table.\n"
            "Best-fit curve:    Tabulated x–y values of the best-fit curve.\n"
            "PTE histogram:     χ² distribution from mock datasets with the\n"
            "                   observed χ² marked; only written when PTE is enabled."
        ), font=("Arial", 9), bg="#FFF8E7", fg="#555", justify=tk.LEFT).pack(anchor="w")

        self._v_o_contour_plot = tk.StringVar(value="./output/contour.pdf")
        self._v_o_bestfit_plot = tk.StringVar(value="./output/bestfit.pdf")
        self._v_o_results      = tk.StringVar(value="./output/results.txt")
        self._v_o_contour_dat  = tk.StringVar(value="./output/posterior.txt")
        self._v_o_bestfit_dat  = tk.StringVar(value="./output/bestfit_curve.txt")
        self._v_o_pte_plot     = tk.StringVar(value="./output/pte_test.pdf")

        for label, var in [
            ("Corner plot (PDF):",    self._v_o_contour_plot),
            ("Best-fit plot (PDF):",  self._v_o_bestfit_plot),
            ("Results summary (.txt):", self._v_o_results),
            ("Posterior grid (.txt):", self._v_o_contour_dat),
            ("Best-fit curve (.txt):", self._v_o_bestfit_dat),
            ("PTE histogram (PDF):",   self._v_o_pte_plot),
        ]:
            self._saveas_field(f, label, var)

        tk.Label(f, text="  The PTE histogram is only written if 'Run PTE test' is enabled in Step 5.",
                 font=("Arial", 9), bg=BG, fg="#888").pack(anchor="w", padx=30)

        self._frames.append(f)

    # ── Step 8: Run ───────────────────────────────────────────────────── #

    def _s_run(self):
        f = self._make_step_frame("Run the Fit",
                                  "Click Run to start.  Output appears in the log below.")
        self._run_btn = ttk.Button(
            f, text="▶   Run polyfit", command=self._launch_run)
        self._run_btn.pack(anchor="w", padx=30, pady=(0, 10))

        lf = tk.LabelFrame(f, text="Log", bg=BG, padx=4, pady=4)
        lf.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 10))
        self._log = scrolledtext.ScrolledText(
            lf, font=("Courier", 9), state=tk.DISABLED,
            bg="#0D1117", fg="#58A6FF",
            insertbackground="white", height=16)
        self._log.pack(fill=tk.BOTH, expand=True)
        self._frames.append(f)

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    def _show_step(self, idx: int):
        for fr in self._frames:
            fr.pack_forget()
        self._frames[idx].pack(fill=tk.BOTH, expand=True)

        last = len(self.STEPS) - 1
        for i, lbl in enumerate(self._sb_labels):
            if i < idx:
                lbl.config(fg=DONE_COL, font=("Arial", 10))
            elif i == idx:
                lbl.config(fg="white", font=("Arial", 10, "bold"))
            else:
                lbl.config(fg=MUTED, font=("Arial", 10))

        self._btn_back.config(state=tk.NORMAL if idx > 0     else tk.DISABLED)
        self._btn_next.config(state=tk.DISABLED if idx == last else tk.NORMAL,
                              text="Next →")
        self._step = idx

    def _go_next(self):
        if self._step < len(self.STEPS) - 1:
            self._show_step(self._step + 1)

    def _go_back(self):
        if self._step > 0:
            self._show_step(self._step - 1)

    # ------------------------------------------------------------------ #
    # State persistence — remember settings between sessions
    # ------------------------------------------------------------------ #

    def _save_state(self):
        cfg = configparser.RawConfigParser()
        cfg["data"] = {"file": self._v_data.get()}
        cfg["covariance"] = {
            "xm":  self._cov_xm.get(),  "xv":  self._cov_xv.get(),  "xf":  self._cov_xf.get(),
            "ym":  self._cov_ym.get(),  "yv":  self._cov_yv.get(),  "yf":  self._cov_yf.get(),
            "xym": self._cov_xym.get(), "xyv": self._cov_xyv.get(), "xyf": self._cov_xyf.get(),
            "ffm": self._cov_ffm.get(), "ffv": self._cov_ffv.get(), "fff": self._cov_fff.get(),
        }
        cfg["prior"] = {
            "p1min": self._v_p1min.get(), "p1max": self._v_p1max.get(),
            "p2min": self._v_p2min.get(), "p2max": self._v_p2max.get(),
            "ninter": self._v_ninter.get(),
        }
        cfg["pte"] = {
            "run_pte": str(self._v_run_pte.get()),
            "n_pte":   self._v_n_pte.get(),
        }
        cfg["plotting"] = {
            "xlabel": self._v_xlabel.get(), "ylabel": self._v_ylabel.get(),
            "p1lbl":  self._v_p1lbl.get(),  "p2lbl":  self._v_p2lbl.get(),
            "lcol":   self._v_lcol.get(),   "dcol":   self._v_dcol.get(),
            "tex":    str(self._v_tex.get()), "zoom": self._v_zoom.get(),
            "model":  self._v_model.get(),
            "model_code": base64.b64encode(
                self._model_code.get("1.0", tk.END).encode()).decode(),
        }
        cfg["output"] = {
            "contour_plot": self._v_o_contour_plot.get(),
            "bestfit_plot": self._v_o_bestfit_plot.get(),
            "results":      self._v_o_results.get(),
            "contour_dat":  self._v_o_contour_dat.get(),
            "bestfit_dat":  self._v_o_bestfit_dat.get(),
            "pte_plot":     self._v_o_pte_plot.get(),
        }
        try:
            with open(STATE_FILE, "w") as fh:
                cfg.write(fh)
        except Exception:
            pass

    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        cfg = configparser.RawConfigParser()
        cfg.read(STATE_FILE)
        try:
            if "data" in cfg:
                self._v_data.set(cfg["data"].get("file", self._v_data.get()))

            if "covariance" in cfg:
                c = cfg["covariance"]
                self._cov_xm.set( c.get("xm",  self._cov_xm.get()))
                self._cov_xv.set( c.get("xv",  self._cov_xv.get()))
                self._cov_xf.set( c.get("xf",  self._cov_xf.get()))
                self._cov_ym.set( c.get("ym",  self._cov_ym.get()))
                self._cov_yv.set( c.get("yv",  self._cov_yv.get()))
                self._cov_yf.set( c.get("yf",  self._cov_yf.get()))
                self._cov_xym.set(c.get("xym", self._cov_xym.get()))
                self._cov_xyv.set(c.get("xyv", self._cov_xyv.get()))
                self._cov_xyf.set(c.get("xyf", self._cov_xyf.get()))
                self._cov_ffm.set(c.get("ffm", self._cov_ffm.get()))
                self._cov_ffv.set(c.get("ffv", self._cov_ffv.get()))
                self._cov_fff.set(c.get("fff", self._cov_fff.get()))

            if "prior" in cfg:
                p = cfg["prior"]
                self._v_p1min.set( p.get("p1min",  self._v_p1min.get()))
                self._v_p1max.set( p.get("p1max",  self._v_p1max.get()))
                self._v_p2min.set( p.get("p2min",  self._v_p2min.get()))
                self._v_p2max.set( p.get("p2max",  self._v_p2max.get()))
                self._v_ninter.set(p.get("ninter", self._v_ninter.get()))

            if "pte" in cfg:
                pt = cfg["pte"]
                self._v_run_pte.set(pt.get("run_pte", "false").lower() == "true")
                self._v_n_pte.set(  pt.get("n_pte", self._v_n_pte.get()))

            if "plotting" in cfg:
                pl = cfg["plotting"]
                self._v_xlabel.set(pl.get("xlabel", self._v_xlabel.get()))
                self._v_ylabel.set(pl.get("ylabel", self._v_ylabel.get()))
                self._v_p1lbl.set( pl.get("p1lbl",  self._v_p1lbl.get()))
                self._v_p2lbl.set( pl.get("p2lbl",  self._v_p2lbl.get()))
                self._v_lcol.set(  pl.get("lcol",   self._v_lcol.get()))
                self._v_dcol.set(  pl.get("dcol",   self._v_dcol.get()))
                self._v_tex.set(   pl.get("tex", "false").lower() == "true")
                self._v_zoom.set(  pl.get("zoom",   self._v_zoom.get()))
                saved_model = pl.get("model", "")
                if saved_model in MODELS:
                    self._v_model.set(saved_model)
                    self._load_model_template(None)
                saved_code_b64 = pl.get("model_code", "")
                if saved_code_b64.strip():
                    saved_code = base64.b64decode(
                        saved_code_b64.encode()).decode()
                    self._model_code.delete("1.0", tk.END)
                    self._model_code.insert(tk.END, saved_code)

            if "output" in cfg:
                o = cfg["output"]
                self._v_o_contour_plot.set(o.get("contour_plot", self._v_o_contour_plot.get()))
                self._v_o_bestfit_plot.set(o.get("bestfit_plot", self._v_o_bestfit_plot.get()))
                self._v_o_results.set(     o.get("results",      self._v_o_results.get()))
                self._v_o_contour_dat.set( o.get("contour_dat",  self._v_o_contour_dat.get()))
                self._v_o_bestfit_dat.set( o.get("bestfit_dat",  self._v_o_bestfit_dat.get()))
                self._v_o_pte_plot.set(    o.get("pte_plot",     self._v_o_pte_plot.get()))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Config helpers
    # ------------------------------------------------------------------ #

    def _cov_str(self, mode: str, value: str, filepath: str) -> str:
        return filepath if mode == "file" else f"{mode} {value}"

    def _write_config(self, path: str):
        cfg = configparser.ConfigParser()
        high_conv = MODELS.get(self._v_model.get(), {}).get("high_conversion", False)

        data_structure: dict = {
            "file_data":       self._v_data.get(),
            "high_conversion": str(high_conv).lower(),
            "X_covariance":    self._cov_str(self._cov_xm.get(),  self._cov_xv.get(),  self._cov_xf.get()),
            "Y_covariance":    self._cov_str(self._cov_ym.get(),  self._cov_yv.get(),  self._cov_yf.get()),
            "XY_covariance":   self._cov_str(self._cov_xym.get(), self._cov_xyv.get(), self._cov_xyf.get()),
        }
        # Only write f0_covariance when non-zero and Skeist model is active
        if high_conv:
            ff_str = self._cov_str(self._cov_ffm.get(), self._cov_ffv.get(), self._cov_fff.get())
            # Avoid writing "rel 0.0" (no-op) unless user explicitly set a file
            ff_val = self._cov_ffv.get().strip()
            if self._cov_ffm.get() == "file" or (ff_val and float(ff_val) != 0.0):
                data_structure["f0_covariance"] = ff_str

        cfg["data_structure"] = data_structure
        cfg["inference"] = {
            "prior_range": (
                f"{self._v_p1min.get()}, {self._v_p1max.get()}, "
                f"{self._v_p2min.get()}, {self._v_p2max.get()}"
            ),
            "run_pte": str(self._v_run_pte.get()).lower(),
            "N_pte":   self._v_n_pte.get(),
        }
        cfg["plotting"] = {
            "plot_line_colour":          self._v_lcol.get(),
            "plot_data_colour":          self._v_dcol.get(),
            "independent_variable_name": self._v_xlabel.get(),
            "dependent_variable_name":   self._v_ylabel.get(),
            "parameter_1_name":          self._v_p1lbl.get(),
            "parameter_2_name":          self._v_p2lbl.get(),
            "contour_plot_zoom":         self._v_zoom.get(),
            "use_tex":                   str(self._v_tex.get()),
        }
        cfg["precision"] = {"N_inter": self._v_ninter.get()}
        cfg["output"] = {
            "file_name_contour_plot":  self._v_o_contour_plot.get(),
            "file_name_best_fit_plot": self._v_o_bestfit_plot.get(),
            "file_name_results":       self._v_o_results.get(),
            "file_name_contour":       self._v_o_contour_dat.get(),
            "file_name_best_fit":      self._v_o_bestfit_dat.get(),
            "pte_output_file":         self._v_o_pte_plot.get(),
        }
        with open(path, "w") as fh:
            cfg.write(fh)

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def _log_line(self, msg: str):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, msg + "\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _launch_run(self):
        self._save_state()
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)
        self._run_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        cfg_path = os.path.join(_ROOT, "_polyfit_gui_tmp.ini")
        try:
            import contextlib, io

            # -- compile and validate model code --
            code = self._model_code.get("1.0", tk.END)
            ns: dict = {}
            exec(compile(code, "<gui-model>", "exec"), ns)  # noqa: S102
            if "model" not in ns or "dmodel" not in ns:
                raise ValueError(
                    "Code must define both 'model(parameters, x)' "
                    "and 'dmodel(parameters, x)'.")
            model_fn    = ns["model"]
            dmodel_fn   = ns["dmodel"]
            dmodel_f0   = ns.get("dmodel_f0", None)   # optional f0-error propagation

            # -- write config --
            self._write_config(cfg_path)

            # -- ensure output directories exist --
            for v in (self._v_o_contour_plot, self._v_o_bestfit_plot,
                      self._v_o_results, self._v_o_contour_dat,
                      self._v_o_bestfit_dat, self._v_o_pte_plot):
                d = os.path.dirname(os.path.abspath(v.get()))
                os.makedirs(d, exist_ok=True)

            # -- settings summary --
            data_path = os.path.abspath(self._v_data.get())
            W = 60
            self._log_line("=" * W)
            self._log_line("  polyfit — settings summary")
            self._log_line("=" * W)
            self._log_line(f"  Data file    : {data_path}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Data file not found:\n  {data_path}\n\n"
                    "Go back to Step 1 and check the path.")
            try:
                fmt = self._detect_data_format(data_path)
            except Exception:
                fmt = "(could not detect)"
            self._log_line(f"  Format       : {fmt}")
            high_conv = MODELS.get(self._v_model.get(), {}).get("high_conversion", False)
            self._log_line(f"  Model        : {self._v_model.get()}")
            self._log_line(f"  High conv.   : {'yes (Skeist)' if high_conv else 'no (Mayo-Lewis / other)'}")
            if "CSV" in fmt:
                self._log_line("  Errors       : read from data file columns")
            else:
                x_err  = self._cov_str(self._cov_xm.get(),  self._cov_xv.get(),  self._cov_xf.get())
                y_err  = self._cov_str(self._cov_ym.get(),  self._cov_yv.get(),  self._cov_yf.get())
                xy_err = self._cov_str(self._cov_xym.get(), self._cov_xyv.get(), self._cov_xyf.get())
                self._log_line(f"  X error      : {x_err}")
                self._log_line(f"  Y error      : {y_err}")
                self._log_line(f"  XY error     : {xy_err}")
                if high_conv:
                    ff_err = self._cov_str(self._cov_ffm.get(), self._cov_ffv.get(), self._cov_fff.get())
                    self._log_line(f"  f0 error     : {ff_err}")
            if dmodel_f0 is not None:
                self._log_line("  dmodel_f0    : defined — f0 uncertainty will be propagated")
            self._log_line(f"  Prior p1     : [{self._v_p1min.get()}, {self._v_p1max.get()}]")
            self._log_line(f"  Prior p2     : [{self._v_p2min.get()}, {self._v_p2max.get()}]")
            self._log_line(f"  Grid N_inter : {self._v_ninter.get()} × {self._v_ninter.get()}"
                           f"  ({int(self._v_ninter.get())**2:,} evaluations)")
            if self._v_run_pte.get():
                self._log_line(f"  PTE test     : {self._v_n_pte.get()} mock datasets")
            self._log_line("-" * W)
            self._log_line("  Running fit — this may take 30–300 s …")
            self._log_line("-" * W + "\n")

            from source.fit_model import FitModel

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fit = FitModel(model_fn, dmodel_fn, cfg_path,
                               derivative_model_f0=dmodel_f0)

            for line in buf.getvalue().splitlines():
                self._log_line(line)

            # -- results summary --
            invF = np.linalg.inv(fit.fisher_at_bf)
            sig0 = np.sqrt(invF[0, 0])
            sig1 = np.sqrt(invF[1, 1])
            corr = invF[0, 1] / (sig0 * sig1)

            try:
                p1_lo = fit.best_fit[0] - fit.p1_cdf_marginal_spline(0.159)
                p1_hi = fit.p1_cdf_marginal_spline(0.841) - fit.best_fit[0]
                p2_lo = fit.best_fit[1] - fit.p2_cdf_marginal_spline(0.159)
                p2_hi = fit.p2_cdf_marginal_spline(0.841) - fit.best_fit[1]
                have_ci = True
            except Exception:
                have_ci = False

            N = fit.number_of_data_points
            chi2_dof = 2.0 * fit.max_func / max(N - 2, 1)

            self._log_line("\n" + "=" * W)
            self._log_line("  Results")
            self._log_line("=" * W)
            self._log_line(f"  Data points  : {N}")
            self._log_line(f"  χ²_min       : {2.0 * fit.max_func:.4g}   "
                           f"(χ²/dof ≈ {chi2_dof:.3g}   — ideally near 1.0)")
            self._log_line("-" * W)
            self._log_line(f"  p1  best fit : {fit.best_fit[0]:.6g}")
            self._log_line(f"      Gaussian : ± {sig0:.4g}   "
                           f"(symmetric, from Fisher matrix)")
            if have_ci:
                self._log_line(f"      True 68 % : −{p1_lo:.4g}  /  +{p1_hi:.4g}   "
                               f"(asymmetric, from marginal posterior)")
            self._log_line(f"  p2  best fit : {fit.best_fit[1]:.6g}")
            self._log_line(f"      Gaussian : ± {sig1:.4g}")
            if have_ci:
                self._log_line(f"      True 68 % : −{p2_lo:.4g}  /  +{p2_hi:.4g}")
            self._log_line("-" * W)
            self._log_line(f"  Correlation  : ρ(p1,p2) = {corr:+.3f}   "
                           f"({'strongly' if abs(corr) > 0.7 else 'weakly'} correlated)")
            self._log_line(f"  Contour 68.3 % threshold : {fit.c1:.5g}")
            self._log_line(f"  Contour 95.4 % threshold : {fit.c2:.5g}")
            self._log_line(f"  Contour 99.7 % threshold : {fit.c3:.5g}")
            self._log_line("-" * W)
            self._log_line("  Interpretation of χ²/dof:")
            if chi2_dof < 0.5:
                self._log_line("    < 0.5 — suspiciously good; errors may be over-estimated.")
            elif chi2_dof < 1.5:
                self._log_line("    ~ 1   — good fit; model and errors are consistent.")
            else:
                self._log_line("    > 1.5 — poor fit; model may be wrong or errors under-estimated.")
            self._log_line("    Run the PTE test (Step 5) for a rigorous assessment.")
            self._log_line("-" * W)

            # -- optional PTE test --
            pte_value = None
            if fit.run_pte:
                self._log_line(f"\n  Running PTE test with {fit.N_pte} mock datasets …")
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):
                    pte_value, _ = fit.run_pte_test(
                        N_mock=fit.N_pte, output_file=fit.pte_output_file)
                self._log_line(f"  PTE = {pte_value:.3f}")
                if pte_value < 0.05:
                    self._log_line("        < 0.05 — poor fit; the model likely does not describe the data.")
                elif pte_value > 0.95:
                    self._log_line("        > 0.95 — suspiciously good; errors may be over-estimated.")
                else:
                    self._log_line("        Acceptable fit (0.05 ≤ PTE ≤ 0.95).")
                self._log_line("-" * W)

            self._log_line("  Output files written:")
            output_vars = [
                ("corner plot     ", self._v_o_contour_plot),
                ("best-fit plot   ", self._v_o_bestfit_plot),
                ("results summary ", self._v_o_results),
                ("posterior grid  ", self._v_o_contour_dat),
                ("best-fit curve  ", self._v_o_bestfit_dat),
            ]
            if fit.run_pte:
                output_vars.append(("PTE histogram   ", self._v_o_pte_plot))
            for label, var in output_vars:
                self._log_line(f"    {label} : {os.path.abspath(var.get())}")
            self._log_line("=" * W)
            self._log_line("  Done.")
            self._log_line("=" * W)

        except Exception as exc:
            import traceback
            self._log_line(f"\n  ERROR: {exc}")
            self._log_line(traceback.format_exc())
        finally:
            self.after(0, lambda: self._run_btn.config(state=tk.NORMAL))
            try:
                os.remove(cfg_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------

def main():
    app = PolyfitGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
