"""
fit_model.py — Bayesian parameter inference for non-linear models.

Overview
--------
FitModel inherits all data-loading logic from ReadData and adds the full
inference pipeline:

    1. Maximum-likelihood estimation (MLE) via Nelder-Mead minimisation of
       the negative log-likelihood (chi_square).
    2. Fisher-matrix Gaussian approximation to the posterior — gives
       symmetric ±sigma error bars.
    3. Normalised 2-D posterior grid evaluated over an automatically
       tightened prior range — gives the exact, potentially asymmetric
       credible regions.
    4. 1-D marginal posteriors and CDF splines — gives true asymmetric
       credible intervals.
    5. All plots and result files written to paths specified in the config.

Statistical model
-----------------
The likelihood is a multivariate Gaussian in residual space:

    -2 ln L = ln|R| + (Y - μ)ᵀ R⁻¹ (Y - μ)

where μ_i = model(θ, x_i) and R is the *effective covariance matrix*
that folds in uncertainties on both x and y via first-order error
propagation (Errors-in-Variables, EVM):

    R = C_YY  -  C_XY · T  -  T · C_XY  +  T · C_XX · T

with T = diag(dmodel/dx evaluated at each data point).

For a plain two-column text file with only y-uncertainties, C_XX = 0 and
C_XY = 0, so R reduces to C_YY.

Supported models
----------------
Any Python callable ``model(parameters, x)`` and ``dmodel(parameters, x)``
may be passed.  For the built-in copolymerization models see
``source/models.py``.  When ``high_conversion = true`` in the config the
data supplied to the model is ``x = [f0_i, X_conv_i]`` rather than a
scalar feed composition.

Reference
---------
R. Reischke, Macromolecular Theory and Simulations 32 (2023) 2200063.
DOI: 10.1002/mats.202200063
"""

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy import optimize
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from source.read_data import ReadData


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _resolve_usetex(want_tex: bool) -> bool:
    """Return True only when *want_tex* is True **and** latex + dvipng are on PATH.

    Matplotlib raises a hard error at render time when usetex=True is set
    but the TeX installation is incomplete.  This guard converts the crash
    into a printed warning and graceful fallback to the built-in MathText
    renderer, which handles ``$...$`` notation without any TeX installation.
    """
    if not want_tex:
        return False
    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        print("Warning: use_tex=True requested but latex/dvipng not found "
              "— falling back to Matplotlib MathText renderer.")
        return False
    return True


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FitModel(ReadData):
    """Bayesian parameter inference for a two-parameter non-linear model.

    Inherits data loading from :class:`ReadData`.  On construction the full
    inference pipeline runs automatically: MLE → Fisher matrix →
    posterior grid → marginals → plots → result files.

    Parameters
    ----------
    model : callable
        Forward model ``model(parameters, x) -> float``.
        *parameters* is a length-2 array ``[p1, p2]``; *x* is either a
        scalar (standard models) or a length-2 array ``[f0, X_conv]`` when
        ``high_conversion = true`` in the config (Skeist model).
    derivative_model : callable
        ``dmodel(parameters, x) -> float`` — ∂model/∂x.  Required for EVM
        error propagation through the effective covariance matrix R.
    derivative_model_f0 : callable or None, optional
        ``dmodel_f0(parameters, x) -> float`` — ∂model/∂f0, the partial
        derivative of the forward model with respect to the initial feed
        composition f0.  Only relevant for the Skeist high-conversion model
        when f0 carries measurement uncertainty (non-zero ``CFF``).  Pass
        ``skeist_deriv_f0`` from ``source.models`` to activate f0 error
        propagation; leave as ``None`` to ignore f0 uncertainty (default).
    config_name : str, optional
        Path to the INI configuration file.  Defaults to ``'input.ini'``.

    Attributes set after construction
    ----------------------------------
    best_fit : ndarray, shape (2,)
        MLE parameter vector ``[p1_hat, p2_hat]``.
    fisher_at_bf : ndarray, shape (2, 2)
        Fisher information matrix evaluated at *best_fit*.
        ``np.linalg.inv(fisher_at_bf)`` is the Gaussian covariance matrix,
        whose diagonal gives the squared symmetric 1sigma errors.
    posterior_grid : ndarray, shape (N_inter, N_inter)
        Normalised posterior evaluated on the ``parameter_x x parameter_y``
        grid.
    p1_cdf_marginal_spline, p2_cdf_marginal_spline : callable
        Inverse-CDF splines for each marginal posterior.  Call with a
        probability (e.g. 0.159, 0.841) to get the corresponding parameter
        value for asymmetric credible intervals.
    c1, c2, c3 : float
        Posterior values at the 68.3 %, 95.4 %, and 99.7 % credible-region
        boundaries (used as contour levels).
    Xfid : ndarray
        x-values used for the saved best-fit curve.
    Ybf : ndarray
        Model predictions at *Xfid* (or at the data points for Skeist runs).
    """

    def __init__(self,
                 model,
                 derivative_model,
                 config_name='input.ini',
                 derivative_model_f0=None):
        super().__init__(config_name=config_name)
        self.model = model
        self.derivative_model = derivative_model
        self.derivative_model_f0 = derivative_model_f0
        self.number_parameters = len(self.prior_range[:, 0])
        self.best_fit = None
        self.max_func = None

        # Initial parameter grid from config prior_range
        self.parameter_x = np.linspace(
            self.prior_range[0, 0], self.prior_range[0, 1], self.N_inter)
        self.parameter_y = np.linspace(
            self.prior_range[1, 0], self.prior_range[1, 1], self.N_inter)

        self.evidence = 1.0  # updated by normalize_posterior()

        # ── Run the full pipeline ────────────────────────────────────────
        self.find_MLE(np.ones(self.number_parameters))
        self.fisher_at_bf = self.fisher(self.best_fit)
        self.update_prior_range()       # tighten grid to ±10sigma around MLE

        self.quantile = None
        self.posterior_grid = None
        self.create_posterior_grid()    # N_inter x N_inter grid, normalised by Simpson

        self.p1_pdf_marginal_spline = None
        self.p2_pdf_marginal_spline = None
        self.p1_cdf_marginal_spline = None
        self.p2_cdf_marginal_spline = None

        self.best_fit_curve_plot()      # writes best-fit plot PDF
        self.find_1d_marginal()         # builds marginal splines (no separate plots)
        self.create_contour_plot()      # combined corner plot: contours + marginals
        self.print_give_result()        # writes results.txt and data files

    # ------------------------------------------------------------------ #
    # Data routing
    # ------------------------------------------------------------------ #

    @property
    def _model_x(self):
        """Per-point inputs passed to ``model`` and ``derivative_model``.

        For standard (low-conversion) models this is identical to
        ``data_X`` — a 1-D array of scalar x values.

        For the Skeist high-conversion model (``high_conversion = True``)
        the model needs both the initial feed composition *f0* and the
        fractional conversion *X_conv* for each data point.  This property
        stacks them into a 2-D array of shape ``(N, 2)`` so that iterating
        over it yields ``[f0_i, X_conv_i]`` rows.
        """
        if self.high_conversion and self.data_f0 is not None:
            return np.column_stack([self.data_f0, self.data_X])
        return self.data_X

    # ------------------------------------------------------------------ #
    # Likelihood and effective covariance
    # ------------------------------------------------------------------ #

    def get_R(self, parameters):
        """Compute the effective covariance matrix R at *parameters*.

        R encodes measurement uncertainties on x, y, and optionally f0 via
        first-order (linear) error propagation through the model:

            R = C_YY  -  C_XY · T_x  -  T_x · C_XY  +  T_x · C_XX · T_x
                       +  T_f0 · C_FF · T_f0

        where T_x  = diag(∂F/∂x)  and  T_f0 = diag(∂F/∂f0).
        The f0 term is included only when ``derivative_model_f0`` was
        supplied and ``CFF`` is non-zero (6-column CSV or config
        ``f0_covariance``).  For independent data points all covariance
        matrices are diagonal, reducing every product to element-wise
        multiplication — O(N²) instead of O(N³).

        Parameters
        ----------
        parameters : array-like, length 2
            Current parameter vector ``[p1, p2]``.

        Returns
        -------
        R : ndarray, shape (N, N)
            Effective covariance matrix.
        """
        diag_x = np.array(
            [self.derivative_model(parameters, x) for x in self._model_x],
            dtype=float)
        R = (self.CYY
             - self.CXY * diag_x
             - self.CXY * diag_x[:, np.newaxis]
             + self.CXX * diag_x[:, np.newaxis] * diag_x)

        if self.derivative_model_f0 is not None and self.CFF is not None:
            diag_f0 = np.array(
                [self.derivative_model_f0(parameters, x) for x in self._model_x],
                dtype=float)
            R = R + self.CFF * diag_f0[:, np.newaxis] * diag_f0

        return R

    def chi_square(self, parameters):
        """Negative log-likelihood (up to a constant), equal to the chi-squared statistic.

        Evaluates:

            χ² = ½ [ln|R| + (Y - μ)ᵀ R⁻¹ (Y - μ)]

        Returns 1e20 (effectively -∞ in posterior space) for any parameter
        vector outside the prior range, enforcing a uniform prior.

        Parameters
        ----------
        parameters : array-like, length 2
            Current parameter vector ``[p1, p2]``.

        Returns
        -------
        float
            χ² value (minimised by :meth:`find_MLE`).
        """
        for i in range(self.number_parameters):
            if (parameters[i] < self.prior_range[i, 0] or
                    parameters[i] > self.prior_range[i, 1]):
                return 1e20
        Y_tilde = self.data_Y - np.array(
            [self.model(parameters, x) for x in self._model_x])
        R = self.get_R(parameters)
        _, logdet = np.linalg.slogdet(R)
        return 0.5 * (logdet + Y_tilde @ np.linalg.solve(R, Y_tilde))

    # ------------------------------------------------------------------ #
    # Numerical derivatives
    # ------------------------------------------------------------------ #

    def partial_derivative_vec_arg(self, func, var=0, point=[]):
        """Numerical partial derivative of *func(parameters)* w.r.t. *parameters[var]*.

        Uses a central-difference formula with dx=1e-5.

        Parameters
        ----------
        func : callable
            Function of the full parameter vector, e.g. ``get_R``.
        var : int
            Index of the parameter to differentiate with respect to.
        point : list
            Parameter vector at which to evaluate the derivative.

        Returns
        -------
        float or ndarray
            Partial derivative (same shape as the output of *func*).
        """
        args = point[:]

        def wraps(x):
            args[var] = x
            return func(args)
        h = 1e-5
        return (wraps(point[var] + h) - wraps(point[var] - h)) / (2.0 * h)

    def partial_derivative(self, func, data, var=0, point=[]):
        """Numerical partial derivative of *func(parameters, data)* w.r.t. *parameters[var]*.

        Uses a central-difference formula with dx=1e-5.
        The *data* argument (a single x value or ``[f0, X_conv]`` pair) is
        held fixed while the selected parameter is varied.

        Parameters
        ----------
        func : callable
            Function with signature ``func(parameters, data)``, e.g.
            ``self.model``.
        data : float or array-like
            Fixed x value (or ``[f0, X_conv]`` for Skeist) for this data point.
        var : int
            Index of the parameter to differentiate with respect to.
        point : list
            Parameter vector at which to evaluate the derivative.

        Returns
        -------
        float
            ∂func/∂parameters[var] at *point* and *data*.
        """
        args = point[:]

        def wraps(x):
            args[var] = x
            return func(args, data)
        h = 1e-5
        return (wraps(point[var] + h) - wraps(point[var] - h)) / (2.0 * h)

    # ------------------------------------------------------------------ #
    # MLE and Fisher matrix
    # ------------------------------------------------------------------ #

    def find_MLE(self, guess):
        """Find the maximum-likelihood estimate via Nelder-Mead minimisation.

        Minimises :meth:`chi_square` (= -ln L up to a constant).  The
        result is stored in ``self.best_fit`` and ``self.max_func``.

        Parameters
        ----------
        guess : array-like, length 2
            Initial parameter guess ``[p1_0, p2_0]``.

        Raises
        ------
        Exception
            If *guess* has the wrong length.
        """
        if len(guess) != self.number_parameters:
            raise Exception("Starting values does not have the right shape.")
        res = minimize(self.chi_square, guess, method='nelder-mead',
                       options={'xatol': 1e-6, 'disp': True})
        # Clip to prior range — Nelder-Mead can step slightly outside the soft
        # 1e20 wall near the boundary; clipping keeps max_func finite.
        self.best_fit = np.clip(res.x,
                                self.prior_range[:, 0],
                                self.prior_range[:, 1])
        self.max_func = self.chi_square(self.best_fit)

    def fisher(self, parameters):
        """Compute the Fisher information matrix at *parameters*.

        The Fisher matrix for a Gaussian likelihood with parameter-dependent
        mean **and** covariance is:

            F_ab = ½ Tr[R⁻¹ ∂R/∂θ_a · R⁻¹ ∂R/∂θ_b]
                   + (∂μ/∂θ_a)ᵀ R⁻¹ (∂μ/∂θ_b)

        Its inverse, evaluated at the MLE, is the Cramér-Rao lower bound on
        parameter variances and gives the Gaussian ±sigma error bars.

        Parameters
        ----------
        parameters : array-like, length 2

        Returns
        -------
        ndarray, shape (2, 2)
            Fisher information matrix.
        """
        dR = np.zeros((2, self.number_of_data_points, self.number_of_data_points))
        dmu = np.zeros((2, self.number_of_data_points))
        R = self.get_R(parameters)
        dR[0] = self.partial_derivative_vec_arg(self.get_R, 0, parameters)
        dR[1] = self.partial_derivative_vec_arg(self.get_R, 1, parameters)
        mx = self._model_x
        for i in range(self.number_of_data_points):
            dmu[0, i] = self.partial_derivative(self.model, mx[i], 0, parameters)
            dmu[1, i] = self.partial_derivative(self.model, mx[i], 1, parameters)
        Rinv = np.linalg.inv(R)
        fisher_matrix = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                fisher_matrix[a, b] = 0.5 * np.trace(
                    Rinv @ dR[a] @ Rinv @ dR[b]
                    + Rinv @ (np.outer(dmu[a], dmu[b]) + np.outer(dmu[b], dmu[a])))
        return fisher_matrix

    # ------------------------------------------------------------------ #
    # Prior range and posterior
    # ------------------------------------------------------------------ #

    def update_prior_range(self):
        """Tighten the parameter grid to ±10sigma (Gaussian) around the MLE.

        The user-supplied prior range is often much wider than needed.
        This method shrinks it to the MLE ± 10 Gaussian standard deviations
        (derived from the Fisher matrix) so the posterior grid concentrates
        its resolution where the probability mass actually lives.
        ``parameter_x`` and ``parameter_y`` are updated accordingly.
        """
        invFisher = np.linalg.inv(self.fisher_at_bf)
        shift = [10.0 * np.sqrt(invFisher[i, i]) for i in range(2)]
        names = ['p1', 'p2']
        for i in range(self.number_parameters):
            sigma_i = shift[i] / 10.0
            lo, hi = self.prior_range[i, 0], self.prior_range[i, 1]
            if self.best_fit[i] - lo < 2.0 * sigma_i:
                print(f"  WARNING: MLE of {names[i]} ({self.best_fit[i]:.4g}) is "
                      f"within 2σ of the prior lower bound ({lo:.4g}). "
                      f"Consider widening prior_range.")
            if hi - self.best_fit[i] < 2.0 * sigma_i:
                print(f"  WARNING: MLE of {names[i]} ({self.best_fit[i]:.4g}) is "
                      f"within 2σ of the prior upper bound ({hi:.4g}). "
                      f"Consider widening prior_range.")
            if self.best_fit[i] - shift[i] > lo:
                self.prior_range[i, 0] = self.best_fit[i] - shift[i]
            if self.best_fit[i] + shift[i] < hi:
                self.prior_range[i, 1] = self.best_fit[i] + shift[i]
        self.parameter_x = np.linspace(
            self.prior_range[0, 0], self.prior_range[0, 1], self.N_inter)
        self.parameter_y = np.linspace(
            self.prior_range[1, 0], self.prior_range[1, 1], self.N_inter)

    def posterior(self, parameters):
        """Normalised posterior at *parameters* (uniform prior on the grid).

        Returns ``evidence x exp(-(χ²(θ) - χ²_min))``.
        ``evidence`` is set by :meth:`create_posterior_grid`; before that
        it is 1.0 and the value is unnormalised.

        Parameters
        ----------
        parameters : array-like, length 2

        Returns
        -------
        float
        """
        return self.evidence * np.exp(-(self.chi_square(parameters) - self.max_func))

    # ------------------------------------------------------------------ #
    # Posterior grid
    # ------------------------------------------------------------------ #

    def create_posterior_grid(self):
        """Evaluate and normalise the posterior on the N_inter x N_inter parameter grid.

        Builds the unnormalised grid exp(-(χ²(θ) - χ²_min)) via O(N_inter²)
        likelihood evaluations, then normalises it in one broadcast pass using
        a 2-D Simpson integration over the already-computed grid values:

            evidence = ∫∫ exp(-Δχ²) dp1 dp2   [Simpson on the grid]
            posterior_grid[i, j] = exp(-Δχ²_ij) / evidence

        This replaces the previous ``dblquad`` call (which re-evaluated the
        likelihood at adaptive interior points) with a single vectorised
        operation at no extra model-evaluation cost.

        Stores the normalised result in ``self.posterior_grid`` (shape
        ``(N_inter, N_inter)``) and updates ``self.evidence``.
        """
        print("creating posterior grid")

        def _eval_row(px):
            row = np.empty(self.N_inter)
            for j, py in enumerate(self.parameter_y):
                row[j] = np.exp(-(self.chi_square([px, py]) - self.max_func))
            return row

        raw = np.empty((self.N_inter, self.N_inter))
        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(_eval_row, px): i
                       for i, px in enumerate(self.parameter_x)}
            for done, fut in enumerate(as_completed(futures), 1):
                raw[futures[fut]] = fut.result()
                print(f"\r  {done}/{self.N_inter} rows", end='', flush=True)
        print()

        inner   = simpson(raw, x=self.parameter_y, axis=1)
        evidence = simpson(inner, x=self.parameter_x)
        self.evidence      = 1.0 / evidence
        self.posterior_grid = raw * self.evidence

    # ------------------------------------------------------------------ #
    # Credible-region contours
    # ------------------------------------------------------------------ #

    def find_contour_root(self, threshold):
        """Residual function for the contour-level bisection search.

        For a candidate posterior threshold value, integrates the portion
        of the posterior that lies above it and returns the difference from
        the target credible-region fraction ``self.quantile``.  Used as the
        zero-finding target in :meth:`find_contour`.

        Parameters
        ----------
        threshold : float
            Candidate posterior level.

        Returns
        -------
        float
            Integrated probability above *threshold* minus ``self.quantile``.
        """
        masked = np.where(self.posterior_grid > threshold, self.posterior_grid, 0.0)
        inner_integral = simpson(masked, x=self.parameter_y, axis=1)
        result = simpson(inner_integral, x=self.parameter_x)
        return result - self.quantile

    def find_contour(self, quantile):
        """Find the posterior level enclosing a given credible-region fraction.

        Uses Brent's method to find the threshold value *c* such that
        ∫∫_{p(θ) > c} p(θ) dθ = *quantile*.

        Parameters
        ----------
        quantile : float
            Target probability mass, e.g. 0.683 for the 68.3 % region.

        Returns
        -------
        float
            Posterior value to use as a contour level when plotting.
        """
        self.quantile = quantile
        max_val = np.amax(self.posterior_grid)
        min_val = np.amin(self.posterior_grid)
        sol = optimize.root_scalar(
            self.find_contour_root,
            bracket=[min_val, max_val],
            method='brentq', rtol=1e-4)
        return sol.root

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #

    def create_contour_plot(self):
        """Write the combined corner plot to ``file_name_contour_plot``.

        Three-panel layout (corner-plot style):

        ┌──────────────┬──────┐
        │  p1 marginal │      │
        ├──────────────┤      │
        │  2-D contour │  p2  │
        │  + ellipses  │ marg │
        └──────────────┴──────┘

        The 2-D panel shows exact posterior contours at 68.3 %, 95.4 %, and
        99.7 % credible regions with Gaussian ellipses overlaid in red.
        The marginal panels show the exact 1-D posteriors (from
        :meth:`find_1d_marginal`) alongside the Gaussian approximation.
        The p2 marginal is rotated 90° so its axis aligns with the 2-D panel.
        """
        covariance = np.linalg.inv(self.fisher_at_bf)
        mean_var = (covariance[0, 0] + covariance[1, 1]) / 2.0
        diff_var = np.sqrt((covariance[0, 0] - covariance[1, 1])**2 / 4.0
                           + covariance[0, 1]**2)
        semi_a = np.sqrt(mean_var + diff_var)
        semi_b = np.sqrt(mean_var - diff_var)
        theta  = np.arctan2(semi_a**2 - covariance[0, 0], covariance[0, 1]) / np.pi * 180.0
        std_p1 = np.sqrt(covariance[0, 0])
        std_p2 = np.sqrt(covariance[1, 1])

        fontsi  = 20
        fontsi2 = 14
        plt.rc('text', usetex=_resolve_usetex(self.use_tex))
        plt.rc('font', family='Arial')
        plt.rcParams['xtick.labelsize'] = '14'
        plt.rcParams['ytick.labelsize'] = '14'

        self.c1 = self.find_contour(0.683)
        self.c2 = self.find_contour(0.954)
        self.c3 = self.find_contour(0.997)

        # Axis limits (same zoom as before)
        x_lo = self.contour_plot_zoom * self.parameter_x[0]
        x_hi = self.parameter_x[-1] / self.contour_plot_zoom
        y_lo = self.contour_plot_zoom * self.parameter_y[0]
        y_hi = self.parameter_y[-1] / self.contour_plot_zoom

        # Marginal PDF values on the grid
        p1_pdf = self.p1_pdf_marginal_spline(self.parameter_x)
        p2_pdf = self.p2_pdf_marginal_spline(self.parameter_y)

        # ── Build the 3-panel corner layout ────────────────────────────
        fig = plt.figure(figsize=(8, 8))
        gs  = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.05,
                               width_ratios=[3, 1], height_ratios=[1, 3])
        ax_p1     = fig.add_subplot(gs[0, 0])
        ax_2d     = fig.add_subplot(gs[1, 0], sharex=ax_p1)
        ax_p2     = fig.add_subplot(gs[1, 1], sharey=ax_2d)
        ax_legend = fig.add_subplot(gs[0, 1])   # top-right: legend only
        ax_legend.axis('off')

        # ── Shared legend in the free top-right corner ──────────────────
        legend_handles = [
            Line2D([0], [0], color=self.plot_line_colour, ls='-',  label='exact'),
            Line2D([0], [0], color='red',                 ls='--', label='Gaussian'),
        ]
        ax_legend.legend(handles=legend_handles, fontsize=fontsi2,
                         frameon=False, loc='center left')

        # ── p1 marginal (top-left) ──────────────────────────────────────
        ax_p1.plot(self.parameter_x, p1_pdf,
                   color=self.plot_line_colour)
        ax_p1.plot(self.parameter_x,
                   self.get_1d_gaussian(self.parameter_x, self.best_fit[0], std_p1),
                   color='red', ls='--')
        ax_p1.set_ylabel('marginal posterior', fontsize=fontsi2)
        ax_p1.set_xlim(x_lo, x_hi)
        ax_p1.set_ylim(bottom=0)
        plt.setp(ax_p1.get_xticklabels(), visible=False)

        # ── 2-D contour (bottom-left) ───────────────────────────────────
        ax_2d.contourf(self.parameter_x, self.parameter_y, self.posterior_grid.T,
                       levels=50, cmap='Blues')
        
        CS = ax_2d.contour(self.parameter_x, self.parameter_y,
                           self.posterior_grid.T,
                           levels=[self.c3, self.c2, self.c1],
                           colors=[self.plot_line_colour] * 3)
        fmt = {}
        for lvl, s in zip(CS.levels, [r'$99.7\%$', r'$95.4\%$', r'$68.3\%$']):
            fmt[lvl] = s
        ax_2d.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
        ax_2d.set_xlabel(self.plot_parameter_1, fontsize=fontsi)
        ax_2d.set_ylabel(self.plot_parameter_2, fontsize=fontsi)
        ax_2d.plot(self.best_fit[0], self.best_fit[1],
                   marker='x', color=self.plot_data_colour)
        ax_2d.set_xlim(x_lo, x_hi)
        ax_2d.set_ylim(y_lo, y_hi)
        for scale in [1.51, 2.48, 3.44]:
            ax_2d.add_patch(Ellipse(
                (self.best_fit[0], self.best_fit[1]),
                width=semi_a * 2 * scale, height=semi_b * 2 * scale,
                angle=theta, edgecolor='red', facecolor='none', ls='-', alpha=1))

        # ── p2 marginal, rotated 90° (bottom-right) ─────────────────────
        ax_p2.plot(p2_pdf, self.parameter_y,
                   color=self.plot_line_colour)
        ax_p2.plot(self.get_1d_gaussian(self.parameter_y, self.best_fit[1], std_p2),
                   self.parameter_y,
                   color='red', ls='--')
        ax_p2.set_xlabel('marginal posterior', fontsize=fontsi2)
        ax_p2.set_xlim(left=0)
        ax_p2.set_ylim(y_lo, y_hi)
        plt.setp(ax_p2.get_yticklabels(), visible=False)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(self.file_name_contour_plot, bbox_inches='tight')
        plt.close(fig)

    def best_fit_curve_plot(self):
        """Write the best-fit curve plot to ``file_name_best_fit_plot``.

        For standard (low-conversion) models: plots a single continuous
        model curve over [0, 1] with data points and error bars.

        For the Skeist high-conversion model: plots one curve per unique
        initial feed composition f0 found in the data, since F_cum depends
        on both f0 and X_conv.

        Also populates ``self.Xfid`` and ``self.Ybf`` used by
        :meth:`print_give_result` to save the curve data to a text file.
        """
        fontsi = 20
        fontsi2 = 16
        plt.tick_params(labelsize=fontsi)
        plt.rc('text', usetex=_resolve_usetex(self.use_tex))
        plt.rc('font', family='Arial')
        plt.rcParams['xtick.labelsize'] = '16'
        plt.rcParams['ytick.labelsize'] = '16'

        fig, ax1 = plt.subplots()
        plt.errorbar(self.data_X, self.data_Y,
                     np.sqrt(np.diagonal(self.CYY)),
                     np.sqrt(np.diagonal(self.CXX)),
                     marker='.', ls='', label=r"data",
                     color=self.plot_data_colour)

        self.Nfid = 500
        if self.high_conversion and self.data_f0 is not None:
            # Skeist: one curve per unique f0, each with a distinct style and label
            X_range   = np.linspace(0.001, 0.999, self.Nfid)
            linestyles = ['-', '--', '-.', ':']
            colors     = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for idx, f0_val in enumerate(np.unique(np.round(self.data_f0, 6))):
                Y_curve = [self.model(self.best_fit, [f0_val, X]) for X in X_range]
                plt.plot(X_range, Y_curve,
                         ls=linestyles[idx % len(linestyles)],
                         color=colors[idx % len(colors)],
                         label=f"$f_0 = {f0_val:.2f}$")
            # Store model predictions at each data point for the output file
            self.Xfid = self.data_X
            self.Ybf  = np.array([self.model(self.best_fit, x) for x in self._model_x])
        else:
            # Standard: single continuous curve from 0 to 1
            self.Xfid = np.linspace(0, 1.0, self.Nfid)
            self.Ybf  = np.array([self.model(self.best_fit, x) for x in self.Xfid])
            plt.plot(self.Xfid, self.Ybf, ls='-',
                     label=r"best fit model", color=self.plot_line_colour)

        plt.xlabel(self.plot_independent_variable, fontsize=fontsi)
        plt.ylabel(self.plot_dependent_variable, fontsize=fontsi)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1),
                   fontsize=fontsi2, frameon=False)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.file_name_best_fit_plot, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------ #
    # 1-D marginals
    # ------------------------------------------------------------------ #

    def get_1d_gaussian(self, x, mu, sigma):
        """Evaluate a 1-D Gaussian PDF at *x*.

        Used to overlay the Gaussian approximation on the exact marginal
        posterior in the 1-D marginal plots.

        Parameters
        ----------
        x : float or ndarray
            Evaluation point(s).
        mu : float
            Mean (= MLE value for the relevant parameter).
        sigma : float
            Standard deviation (= Gaussian 1sigma error from Fisher matrix).

        Returns
        -------
        float or ndarray
        """
        return (1.0 / np.sqrt(2.0 * np.pi * sigma**2)
                * np.exp(-0.5 * (x - mu)**2 / sigma**2))

    def find_1d_marginal(self):
        """Compute 1-D marginal posteriors and build CDF splines.

        Marginals are obtained by integrating the 2-D posterior grid along
        each parameter axis (Simpson's rule).  A fine grid CDF is computed
        via cumulative trapezoidal integration and normalised to [0, 1].

        The inverse-CDF splines (``p1_cdf_marginal_spline``,
        ``p2_cdf_marginal_spline``) map a probability p to the corresponding
        parameter value — call with 0.159 / 0.841 to get the true
        asymmetric -1sigma / +1sigma credible interval boundaries.

        The 1-D marginal curves are rendered inside the combined corner plot
        produced by :meth:`create_contour_plot`; no separate plot files are
        written here.
        """
        # Marginal PDFs by integrating the grid along the other axis
        p1_marginal = simpson(self.posterior_grid, x=self.parameter_y, axis=-1)
        p2_marginal = simpson(self.posterior_grid, x=self.parameter_x, axis=-2)
        self.p1_pdf_marginal_spline = interp1d(
            self.parameter_x, p1_marginal, fill_value="extrapolate")
        self.p2_pdf_marginal_spline = interp1d(
            self.parameter_y, p2_marginal, fill_value="extrapolate")

        # CDF on a fine grid (10x resolution) then interpolate back to grid points
        n_fine = 10 * self.N_inter
        fine_x = np.linspace(self.parameter_x[0], self.parameter_x[-1], n_fine)
        fine_y = np.linspace(self.parameter_y[0], self.parameter_y[-1], n_fine)
        p1_cdf_fine = cumulative_trapezoid(
            self.p1_pdf_marginal_spline(fine_x), fine_x, initial=0)
        p2_cdf_fine = cumulative_trapezoid(
            self.p2_pdf_marginal_spline(fine_y), fine_y, initial=0)
        p1_cdf_fine /= p1_cdf_fine[-1]
        p2_cdf_fine /= p2_cdf_fine[-1]
        p1_cdf_marginal = np.interp(self.parameter_x, fine_x, p1_cdf_fine)
        p2_cdf_marginal = np.interp(self.parameter_y, fine_y, p2_cdf_fine)
        p1_cdf_marginal[-1] = 1.0
        p2_cdf_marginal[-1] = 1.0

        # Inverse-CDF splines: CDF value → parameter value
        self.p1_cdf_marginal_spline = interp1d(
            p1_cdf_marginal, self.parameter_x, fill_value="extrapolate")
        self.p2_cdf_marginal_spline = interp1d(
            p2_cdf_marginal, self.parameter_y, fill_value="extrapolate")

    # ------------------------------------------------------------------ #
    # Output files
    # ------------------------------------------------------------------ #

    def print_give_result(self):
        """Write all result files to the paths specified in the config.

        Writes three files:

        ``file_name_results``
            Human-readable summary: MLE values, symmetric Gaussian 1sigma
            errors, true asymmetric 68% credible intervals from the
            marginal CDFs, and the posterior contour levels.

        ``file_name_best_fit``
            Two-column text (x, model(x)) for the best-fit curve.
            For Skeist runs: three-column text (f0, X_conv, F_cum_model)
            giving predictions at each observed data point.

        ``file_name_contour``
            Three-column text (p1, p2, posterior) over the full grid —
            suitable for re-plotting in any software.
        """
        invFisher = np.linalg.inv(self.fisher_at_bf)
        std_p1 = np.sqrt(invFisher[0, 0])
        std_p2 = np.sqrt(invFisher[1, 1])

        with open(self.file_name_results, 'w') as f:
            f.write("### Results\n\n")
            f.write(f"The best fit values are: (p1,p2) = "
                    f"({self.best_fit[0]}, {self.best_fit[1]})\n\n")
            f.write(f"Corresponding symmetric one sigma Gaussian errors are: "
                    f"({std_p1},{std_p2})\n\n")
            f.write(f"Corresponding real left errors are: -("
                    f"{self.best_fit[0] - self.p1_cdf_marginal_spline(0.16)},"
                    f"{self.best_fit[1] - self.p2_cdf_marginal_spline(0.16)})\n\n")
            f.write(f"Corresponding real right errors are: +("
                    f"{self.p1_cdf_marginal_spline(0.84) - self.best_fit[0]},"
                    f"{self.p2_cdf_marginal_spline(0.84) - self.best_fit[1]})\n\n")
            f.write("The contour levels to plot the confidence intervals from the "
                    "posterior text files are: "
                    f"{self.c1}, {self.c2}, {self.c3} "
                    "for 68.3%, 95.4%, and 99.7%, respectively\n\n")

        with open(self.file_name_best_fit, 'w') as f:
            if self.high_conversion and self.data_f0 is not None:
                f.write("### Best fit curve (Skeist high-conversion)\n")
                f.write("### f0   X_conv   F_cum_model\n")
                np.savetxt(f, np.column_stack([self.data_f0, self.Xfid, self.Ybf]))
            else:
                f.write("### Best fit curve\n")
                f.write("### independent variable   dependent variable\n")
                np.savetxt(f, np.column_stack([self.Xfid, self.Ybf]))

        with open(self.file_name_contour, 'w') as f:
            f.write("### 2D posterior distribution\n")
            f.write("### parameter_1    parameter_2    posterior\n")
            px, py = np.meshgrid(self.parameter_x, self.parameter_y, indexing='ij')
            np.savetxt(f, np.column_stack([px.ravel(), py.ravel(),
                                           self.posterior_grid.ravel()]))

    # ------------------------------------------------------------------ #
    # Goodness-of-fit
    # ------------------------------------------------------------------ #

    def run_pte_test(self, N_mock=500, output_file='pte_test.pdf', seed=None):
        """Run a Monte Carlo PTE (Probability To Exceed) goodness-of-fit test.

        Generates *N_mock* synthetic datasets by drawing from the multivariate
        Gaussian implied by the best-fit model and effective covariance matrix,
        re-fits each one, and accumulates the distribution of best-fit χ²
        values.  The PTE is the fraction of mock realisations whose χ²_min
        exceeds the data value:

            PTE = #{χ²_mock ≥ χ²_data} / N_mock

        A PTE near 0.5 indicates a good fit.  Values < 0.05 (data χ² is
        unexpectedly large) or > 0.95 (unexpectedly small, suggesting the
        uncertainties are over-estimated) signal problems.

        The mock fitting starts from the best-fit parameter values with loose
        tolerances so each re-fit is fast.  For N ~ 20 data points and
        N_mock = 500 the test typically takes a few minutes.

        Parameters
        ----------
        N_mock : int, optional
            Number of Monte Carlo realisations (default 500).
        output_file : str, optional
            Path for the saved histogram PDF (default ``'pte_test.pdf'``).
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        pte : float
            PTE value in [0, 1].
        chi2_mock : ndarray, shape (N_mock,)
            Best-fit χ² values from all mock realisations.
        """
        rng = np.random.default_rng(seed)

        # Model predictions and covariance at the MLE
        mu = np.array([self.model(self.best_fit, x) for x in self._model_x])
        R  = self.get_R(self.best_fit)

        chi2_data = self.max_func   # χ²_min of the real data
        Y_orig    = self.data_Y.copy()

        chi2_mock = np.empty(N_mock)
        print(f"Running PTE test with {N_mock} mock realisations …")
        for k in range(N_mock):
            if (k + 1) % 100 == 0:
                print(f"  {k + 1} / {N_mock}")
            self.data_Y = rng.multivariate_normal(mu, R)
            res = minimize(self.chi_square, self.best_fit, method='nelder-mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            chi2_mock[k] = self.chi_square(res.x)

        # Restore original data
        self.data_Y = Y_orig

        pte = float(np.mean(chi2_mock >= chi2_data))
        print(f"PTE = {pte:.3f}  (χ²_data = {chi2_data:.4f}, "
              f"median χ²_mock = {np.median(chi2_mock):.4f})")

        # ── Histogram plot ───────────────────────────────────────────────
        fontsi  = 20
        fontsi2 = 16
        plt.rc('text', usetex=_resolve_usetex(self.use_tex))
        plt.rc('font', family='Arial')
        plt.rcParams['xtick.labelsize'] = '14'
        plt.rcParams['ytick.labelsize'] = '14'

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(chi2_mock, bins=max(20, N_mock // 25), density=True,
                color=self.plot_line_colour, alpha=0.6,
                label=f'mock realisations ($N={N_mock}$)')
        ax.axvline(chi2_data, color=self.plot_data_colour, lw=1, ls=':',
                   label=f'data  (PTE $= {pte:.3f}$)')
        ax.set_xlabel(r'$\chi^2_{\mathrm{min}}$', fontsize=fontsi)
        ax.set_ylabel('probability density', fontsize=fontsi)
        ax.set_ylim(bottom=0)
        ax.legend(loc='best', fontsize=fontsi2, frameon=False)
        fig.tight_layout()
        fig.savefig(output_file)
        plt.close(fig)

        return pte, chi2_mock
