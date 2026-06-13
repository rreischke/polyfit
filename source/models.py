"""
models.py — Built-in forward models for copolymerization kinetics.

Each model is a pair of callables with the signatures expected by FitModel:

    model(parameters, x)   -> float   — forward prediction
    dmodel(parameters, x)  -> float   — d(model)/dx, for EVM error propagation

Convention (both models)
------------------------
    parameters[0] = r2  — reactivity ratio of monomer M2
    parameters[1] = r1  — reactivity ratio of monomer M1
    x  (or f)           = mole fraction of M2 in the feed

Choosing between models
-----------------------
Mayo-Lewis  — valid when cumulative conversion X < ~5 %.  The feed
              composition barely changes during the reaction, so the
              instantaneous composition is a good approximation of the
              cumulative copolymer composition.

Skeist      — required when X > ~5 %.  The feed drifts as the more
              reactive monomer is consumed faster; the cumulative
              composition is obtained by integrating the Mayo-Lewis
              equation over the full conversion path.

Data format for the Skeist model
---------------------------------
The independent variable is cumulative conversion X_cum, not feed
composition f.  Because each experiment starts from a different initial
feed f0, the data must supply both values per point:

    x = [f0, X_cum]   (a length-2 array per data point)

The ReadData loader produces this automatically when
``high_conversion = true`` is set in the ``[data_structure]`` config
section and a 4- or 5-column CSV file is used (first column = f0).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid


# ---------------------------------------------------------------------------
# Mayo-Lewis instantaneous model  (low conversion, X < ~5 %)
# ---------------------------------------------------------------------------

def mayo_lewis(parameters, f):
    """Instantaneous copolymer composition F as a function of feed composition f.

    Implements the Mayo-Lewis terminal model:

        F(f) = (r2·f² + f·(1-f)) / (r2·f² + 2f·(1-f) + r1·(1-f)²)

    where F and f are both mole fractions of M2.  The equation describes the
    composition of the copolymer formed *instantaneously* at feed composition f,
    assuming only terminal-unit effects (i.e. reactivity depends only on the
    last inserted monomer, not on earlier units in the chain).

    Valid when cumulative conversion X < ~5 %.  At higher conversions use
    :func:`skeist` instead.

    Parameters
    ----------
    parameters : array-like, length 2
        ``[r2, r1]`` — reactivity ratios.  r_i < 1 means monomer i prefers
        to cross-propagate; r_i > 1 means it prefers self-propagation.
    f : float
        Mole fraction of M2 in the feed (0 < f < 1).

    Returns
    -------
    float
        Mole fraction of M2 in the instantaneously formed copolymer.
    """
    r2, r1 = parameters[0], parameters[1]
    num = r2 * f**2 + f * (1 - f)
    den = r2 * f**2 + 2 * f * (1 - f) + r1 * (1 - f)**2
    return num / den


def mayo_lewis_deriv(parameters, f):
    """Analytical derivative dF/df of the Mayo-Lewis model.

    Obtained by applying the quotient rule to the Mayo-Lewis equation.
    Used by FitModel to populate the diagonal of the T matrix in the
    effective covariance matrix R = C_YY - C_XY·T - T·C_XY + T·C_XX·T.

    Parameters
    ----------
    parameters : array-like, length 2
        ``[r2, r1]`` — reactivity ratios.
    f : float
        Mole fraction of M2 in the feed.

    Returns
    -------
    float
        dF/df evaluated at *f*.
    """
    r2, r1 = parameters[0], parameters[1]
    den  = r2 * f**2 + 2 * f * (1 - f) + r1 * (1 - f)**2
    dden = 2 * f * r2 + 2 - 4 * f - 2 * r1 * (1 - f)
    return ((2 * r2 * f - 2 * f + 1) / den
            - (r2 * f**2 + f * (1 - f)) * dden / den**2)


# ---------------------------------------------------------------------------
# Skeist integrated model  (high conversion, X > ~5 %)
# ---------------------------------------------------------------------------

def _skeist_find_f_final(parameters, f0, X_cum, n=400):
    """Solve the Skeist integral equation for the final feed composition f.

    The Skeist equation relates the change in feed composition to cumulative
    conversion by integrating the Mayo-Lewis equation:

        ∫[f0 → f_final] df' / (F(f') - f') = ln(1 - X_cum)

    Instead of calling a quadrature routine inside a root-finder
    the integrand is evaluated on a fixed grid of *n* points in one vectorised
    pass, the cumulative integral is built with the trapezoid rule, and the
    target value ln(1-X) is located by linear interpolation.

    Grid direction
    --------------
    The cumulative integral from f0 is monotonically decreasing in both cases:

    - F(f0) > f0 → M2 consumed faster → f decreases → grid runs f0 → 0⁺.
    - F(f0) < f0 → M1 consumed faster → f increases → grid runs f0 → 1⁻.

    Parameters
    ----------
    parameters : array-like, length 2
        ``[r2, r1]`` — reactivity ratios.
    f0 : float
        Initial feed mole fraction of M2.
    X_cum : float
        Cumulative monomer conversion (0 < X_cum < 1).
    n : int, optional
        Number of grid points (default 400).

    Returns
    -------
    float
        Feed mole fraction of M2 at the end of the reaction, f_final.
        Clamped to the grid edge if conversion exceeds the integrated range.
    """
    target = np.log(1.0 - X_cum)

    if mayo_lewis(parameters, f0) >= f0:
        f_grid = np.linspace(f0, 1e-6, n)
    else:
        f_grid = np.linspace(f0, 1.0 - 1e-6, n)

    denom = mayo_lewis(parameters, f_grid) - f_grid
    with np.errstate(divide='ignore', invalid='ignore'):
        integrand = 1.0 / denom

    bad = ~np.isfinite(integrand)
    if np.any(bad):
        cut = int(np.argmax(bad))
        if cut == 0:
            return f0 
        f_grid    = f_grid[:cut]
        integrand = integrand[:cut]

    cumulative = cumulative_trapezoid(integrand, f_grid, initial=0)

    if target < cumulative[-1]:
        return float(f_grid[-1]) 
    return float(np.interp(target, cumulative[::-1], f_grid[::-1]))


def skeist(parameters, x):
    """Cumulative copolymer composition via the Skeist integrated equation.

    The cumulative composition F_cum is the average composition of all
    copolymer formed from the start of the reaction up to conversion X_cum.
    It is derived from the Skeist integral via the monomer mass balance:

        F_cum = (f0 - f_final · (1 - X_cum)) / X_cum

    where f_final is obtained by solving the Skeist integral numerically
    (see :func:`_skeist_find_f_final`).

    Use this model when cumulative conversion X > ~5 %.  At X → 0 the
    result converges to the instantaneous Mayo-Lewis value.

    Parameters
    ----------
    parameters : array-like, length 2
        ``[r2, r1]`` — reactivity ratios.
    x : array-like, length 2
        ``x[0] = f0``    — initial feed mole fraction of M2.
        ``x[1] = X_cum`` — cumulative monomer conversion (0 < X_cum < 1).

    Returns
    -------
    float
        Cumulative copolymer mole fraction of M2, F_cum.
    """
    f0, X = float(x[0]), float(x[1])
    if X <= 1e-6:
        return mayo_lewis(parameters, f0)
    f_final = _skeist_find_f_final(parameters, f0, X)
    return (f0 - f_final * (1.0 - X)) / X


def skeist_deriv_f0(parameters, x):
    """Analytical derivative dF_cum/df0 (w.r.t. the initial feed composition).

    Differentiating the Skeist integral w.r.t. f0 via the Leibniz rule:

        d/df0 ∫[f0→f_final] g(f') df' = 0   (integral equals ln(1−X) = const.)
        g(f_final) · df_final/df0 − g(f0) = 0
        df_final/df0 = (F(f_final) − f_final) / (F(f0) − f0)

    Substituting into d(F_cum)/df0 from the mass balance:

        dF_cum/df0 = [1 − (F(f_final) − f_final) / (F(f0) − f0) · (1−X)] / X

    This quantity is needed for EVM error propagation when the initial feed
    composition f0 carries measurement uncertainty σ_f0.  The contribution to
    the effective covariance matrix is:

        ΔR_ij = (∂F_cum/∂f0)_i · C_F0F0_ij · (∂F_cum/∂f0)_j

    At X → 0 the expression reduces to mayo_lewis_deriv(parameters, f0).
    At the azeotrope F(f0) = f0, f_final = f0 for all X, so F_cum = f0 and
    dF_cum/df0 = 1.

    Parameters
    ----------
    parameters : array-like, length 2
        ``[r2, r1]`` — reactivity ratios.
    x : array-like, length 2
        ``x[0] = f0``    — initial feed mole fraction of M2.
        ``x[1] = X_cum`` — cumulative monomer conversion.

    Returns
    -------
    float
        dF_cum/df0 evaluated at *x*.
    """
    f0, X = float(x[0]), float(x[1])
    if X <= 1e-6:
        return mayo_lewis_deriv(parameters, f0)
    F_f0 = mayo_lewis(parameters, f0)
    denom_f0 = F_f0 - f0
    if abs(denom_f0) < 1e-12:
        return 1.0  # at the azeotrope: f_final = f0 always, F_cum = f0
    f_final = _skeist_find_f_final(parameters, f0, X)
    F_ff = mayo_lewis(parameters, f_final)
    return (1.0 - (F_ff - f_final) / denom_f0 * (1.0 - X)) / X


def skeist_deriv(parameters, x):
    """Analytical derivative dF_cum/dX_cum.

    Derived by differentiating the mass balance and the Skeist integral
    simultaneously with respect to X.

    From the mass balance  F_cum = (f0 - f_final·(1-X)) / X  and the
    implicit derivative of the Skeist integral:

        df_final/dX = -(F(f_final) - f_final) / (1 - X)

    Substituting and simplifying using the mass balance yields:

        dF_cum/dX = (F(f_final) - F_cum) / X

    where F(f_final) = mayo_lewis(parameters, f_final) is the instantaneous
    copolymer composition at the current (evolved) feed composition.

    At X → 0 the expression is 0/0; applying L'Hôpital's rule gives the
    limit  F'(f0)·(f0 - F(f0)) / 2.

    Parameters
    ----------
    parameters : array-like, length 2
        ``[r2, r1]`` — reactivity ratios.
    x : array-like, length 2
        ``x[0] = f0``    — initial feed mole fraction of M2.
        ``x[1] = X_cum`` — cumulative monomer conversion.

    Returns
    -------
    float
        dF_cum/dX_cum evaluated at *x*.
    """
    f0, X = float(x[0]), float(x[1])
    if X <= 1e-6:
        F0 = mayo_lewis(parameters, f0)
        return mayo_lewis_deriv(parameters, f0) * (f0 - F0) / 2
    f_final = _skeist_find_f_final(parameters, f0, X)
    F_inst = mayo_lewis(parameters, f_final)
    F_cum = (f0 - f_final * (1.0 - X)) / X
    return (F_inst - F_cum) / X
