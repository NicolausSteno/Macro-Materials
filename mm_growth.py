"""
mm_growth.py
============
The fixed-saving capital-material extension of Chapter 2, Section 6.

Implements and verifies:
  * the stationary capital stock k*                (eq. ext_kstar)
  * the 3x3 Jacobian in (k, u, d)                  (eq. ext_jacobian)
  * the decomposition a_1 = tau_M - i_k, a_2 = Pi_M - i_k tau_M - F_M,
    a_3 = -i_k Pi_M - G_M                          (eqs. ext_a1_decomp - ext_a3_decomp)
  * the Schur identity a_3 = (1-alpha-vartheta nu)(n+delta_K) Pi_M
                                                   (eq. ext_a3_schur)
  * Pi_M = (n+lam)(n+gam)(1-chi*)/(1-rho sigma)    (eq. ext_PiM_chi)
  * the closed form i_k = (n+delta_K)[alpha + vartheta nu (1-chi*)/(1-rho sigma) - 1]
                                                   (eq. ext_ik_closed)
  * the Hawkins-Simon / Metzler leading principal minors (eq. ext_D2)
  * the growth-adjusted ratios B_U(g_s), B_D(g_s) and chi(g_s)
                                                   (eqs. ext_au_g - ext_chi_g)
  * the balanced growth rate on the knife edge     (eq. ext_gk_equation)
  * the detrended planar Jacobian on the knife edge (eq. app_Jtilde)
  * the comparative-statics elasticities            (eqs. ext_dlnk_dchi - ext_dgk_dz)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import brentq

from mm_core import MaterialParams, CU_GAMMA


@dataclass(frozen=True)
class TechParams:
    """Illustrative technology block of the growth extension."""
    alpha: float = 0.35
    nu: float = 0.05
    theta_v: float = 0.6      # vartheta
    delta_K: float = 0.05
    iota: float = 0.25        # saving rate
    A_Y: float = 1.0
    B: float = 0.5
    b_P: float = 0.16

    @property
    def varsigma(self) -> float:
        return 1.0 - self.alpha - self.nu

    @property
    def returns(self) -> float:
        """alpha + vartheta * nu, the reduced-form capital elasticity."""
        return self.alpha + self.theta_v * self.nu


TECH = TechParams()


# ----------------------------------------------------------------------
# Growth-adjusted stock-flow ratios
# ----------------------------------------------------------------------

def BU(m: MaterialParams, g: float = 0.0) -> float:
    return (1 - m.sigma) * m.kappa / (m.n + m.lam + g)


def BD(m: MaterialParams, g: float = 0.0) -> float:
    return m.q * (1 - m.rho) * (m.lam * BU(m, g) + m.sigma) / (m.n + m.gamma + g)


def chi_g(m: MaterialParams, g: float = 0.0) -> float:
    """Growth-adjusted circularity chi(g_s), eq. (ext_chi_g)."""
    return m.rho * m.sigma + m.rho * m.lam * BU(m, g) + m.gamma * BD(m, g)


def dchi_dg(m: MaterialParams, g: float = 0.0) -> float:
    """chi'(g_s) < 0, from eqs. (ext_auprime_g)-(ext_adprime_g)."""
    bu, bd = BU(m, g), BD(m, g)
    dbu = -bu / (m.n + m.lam + g)
    dbd = (m.q * (1 - m.rho)
           * (m.lam * dbu * (m.n + m.gamma + g) - (m.lam * bu + m.sigma))
           / (m.n + m.gamma + g) ** 2)
    return m.rho * m.lam * dbu + m.gamma * dbd


# ----------------------------------------------------------------------
# Stationary equilibrium
# ----------------------------------------------------------------------

def Mcal(m: MaterialParams, t: TechParams, chi: float | None = None) -> float:
    """Reduced-form technology multiplier M(b_P, chi), eq. (ext_Mcal)."""
    chi = chi_g(m, 0.0) if chi is None else chi
    return (t.A_Y * (1 - t.b_P) ** t.alpha
            * ((1 - m.sigma) * t.B * t.b_P ** t.theta_v / (1 - chi)) ** t.nu)


def k_star(m: MaterialParams, t: TechParams) -> float:
    """Positive stationary capital stock, eq. (ext_kstar)."""
    assert t.returns < 1, "alpha + vartheta*nu must be < 1 for a stationary regime"
    return (t.iota * Mcal(m, t) / (m.n + t.delta_K)) ** (1 / (1 - t.returns))


def stationary(m: MaterialParams, t: TechParams) -> dict:
    """Full stationary equilibrium (k*, u*, d*, s*, y*)."""
    chi = chi_g(m, 0.0)
    k = k_star(m, t)
    s = t.B * (t.b_P * k) ** t.theta_v / (1 - chi)
    return dict(k=k, s=s, u=BU(m) * s, d=BD(m) * s,
                y=Mcal(m, t) * k ** t.returns, chi=chi,
                p=(1 - chi) * s, x=(1 - m.sigma) * s)


# ----------------------------------------------------------------------
# Local stability
# ----------------------------------------------------------------------

def jacobian3(m: MaterialParams, t: TechParams) -> np.ndarray:
    """Jacobian of the (k, u, d) system, eq. (ext_jacobian)."""
    st = stationary(m, t)
    k, s, y, x = st["k"], st["s"], st["y"], st["x"]
    den = 1 - m.rho * m.sigma
    s_k = t.B * t.theta_v * t.b_P ** t.theta_v * k ** (t.theta_v - 1) / den
    s_u = m.rho * m.lam / den
    s_d = m.gamma / den
    y_k = t.alpha * y / ((1 - t.b_P) * k) * (1 - t.b_P)   # d y / d k at fixed b_P
    y_k = t.alpha * y / k
    y_x = t.nu * y / x
    cU, cD = (1 - m.sigma) * m.kappa, m.q * (1 - m.rho)
    return np.array([
        [t.iota * (y_k + y_x * (1 - m.sigma) * s_k) - (m.n + t.delta_K),
         t.iota * y_x * (1 - m.sigma) * s_u,
         t.iota * y_x * (1 - m.sigma) * s_d],
        [cU * s_k, cU * s_u - (m.n + m.lam), cU * s_d],
        [cD * m.sigma * s_k, cD * (m.lam + m.sigma * s_u),
         cD * m.sigma * s_d - (m.n + m.gamma)],
    ])


def routh_hurwitz(J: np.ndarray) -> dict:
    """Characteristic coefficients and the four Routh-Hurwitz conditions."""
    a1 = -np.trace(J)
    minors = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
              + J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]
              + J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
    a2 = minors
    a3 = -np.linalg.det(J)
    return dict(a1=a1, a2=a2, a3=a3, RH4=a1 * a2 - a3,
                stable=bool(a1 > 0 and a2 > 0 and a3 > 0 and a1 * a2 - a3 > 0),
                eigenvalues=np.linalg.eigvals(J))


def decomposition(m: MaterialParams, t: TechParams) -> dict:
    """The structural decomposition of the Routh-Hurwitz coefficients."""
    st = stationary(m, t)
    chi, den = st["chi"], 1 - m.rho * m.sigma
    cU, cD = (1 - m.sigma) * m.kappa, m.q * (1 - m.rho)
    mU, mD = m.n + m.lam, m.n + m.gamma
    s_u, s_d = m.rho * m.lam / den, m.gamma / den

    tau_M = mU + mD - cU * s_u - cD * m.sigma * s_d
    Pi_M = mU * mD - cU * mD * s_u - cD * (m.sigma * mU + cU * m.lam) * s_d
    Pi_M_closed = mU * mD * (1 - chi) / den

    i_k = (m.n + t.delta_K) * (t.alpha + t.theta_v * t.nu * (1 - chi) / den - 1)
    f_k = (m.n + t.delta_K) * t.theta_v * t.nu * (1 - chi) / den
    F_M = f_k * (cU * s_u + cD * m.sigma * s_d)
    G_M = f_k * (cU * mD * s_u + cD * (m.sigma * mU + cU * m.lam) * s_d)
    G_M_closed = Pi_M * (m.n + t.delta_K) * t.theta_v * t.nu * (chi - m.rho * m.sigma) / den

    d2 = (-i_k) * (mU - cU * s_u) - f_k * cU * s_u
    d2_alt = (m.n + t.delta_K) * (1 - t.alpha) * (mU - cU * s_u) - f_k * mU

    return dict(tau_M=tau_M, Pi_M=Pi_M, Pi_M_closed=Pi_M_closed,
                i_k=i_k, f_k=f_k, F_M=F_M, G_M=G_M, G_M_closed=G_M_closed,
                a1=tau_M - i_k, a2=Pi_M - i_k * tau_M - F_M, a3=-i_k * Pi_M - G_M,
                a3_schur=(1 - t.returns) * (m.n + t.delta_K) * Pi_M,
                minor1=-i_k, minor2=d2, minor2_alt=d2_alt)


# ----------------------------------------------------------------------
# Generalised investment schedule
# ----------------------------------------------------------------------

def regime_code(a1: float, a2: float, a3: float) -> int:
    """Classify a root configuration of eq. (ext_charpoly).

    0 stable, monotone      all Re < 0, all roots real
    1 stable, oscillatory   all Re < 0, one complex pair -- damped approach
    2 saddle, monotone      one positive real root, the other two real negative
    3 saddle, oscillatory   one positive real root, a damped complex pair
    4 explosive oscillation unstable complex pair with a negative real root
    5 repeller              all Re > 0

    Codes 4 and 5 are the third and fourth rows of Table (ext_dynamic_regimes).
    Code 4 needs a_3 > 0: the product of the roots is -a_3, so a negative real
    root forces a_3 > 0, which by eq. (ext_ik_threshold) means i_k < -G_M/Pi_M.
    Growing oscillations therefore live strictly on the diminishing-returns side
    of the plane, while for i_k > 0 the complex pair is always damped and the
    instability is the real saddle root.  Code 5 needs a_2 > 0 together with
    a_1 < 0 and a_1a_2 < a_3, which for positive Pi_M, F_M, G_M and tau_M > 0
    is empty: a_2 > 0 gives Pi_M - F_M + i_k^2 > i_k^2, and a_1a_2 < a_3 then
    requires tau_M above (Pi_M - F_M)/i_k + i_k > i_k, contradicting a_1 < 0.
    """
    ev = np.roots([1.0, a1, a2, a3])
    cplx = bool(np.abs(ev.imag).max() > 1e-14)
    npos = int((ev.real > 1e-15).sum())
    if npos == 0:
        return 1 if cplx else 0
    if npos == 3:
        return 5
    if cplx and npos == 2:
        return 4
    return 3 if cplx else 2


REGIME_NAMES = ["stable, monotone", "stable, damped oscillation",
                "saddle, monotone", "saddle, oscillatory",
                "explosive oscillation", "repeller"]


def hopf_locus(Pi_M: float, F_M: float, G_M: float, i_k):
    """Material damping tau_M on the Hopf boundary a_1 a_2 = a_3.

    Substituting eqs. (ext_a1_decomp)-(ext_a3_decomp) into a_1a_2 = a_3 gives
    the quadratic i_k tau^2 - tau(Pi_M - F_M + i_k^2) - (i_k F_M + G_M) = 0.
    Returns the positive root at which all three coefficients are positive, or
    NaN where the boundary does not exist; it exists only for
    i_k < -G_M/F_M.
    """
    out = []
    for i in np.atleast_1d(np.asarray(i_k, float)):
        if i == 0.0:
            out.append(np.nan)
            continue
        r = np.roots([i, -(Pi_M - F_M + i * i), -(i * F_M + G_M)])
        r = np.sort(r[np.abs(r.imag) < 1e-14].real)
        r = r[r > 0]
        ok = [x for x in r
              if (x - i) > 0 and (Pi_M - i * x - F_M) > 0 and (-i * Pi_M - G_M) > 0]
        out.append(ok[0] if ok else np.nan)
    return np.array(out)


def returns_coeffs(m: MaterialParams, t: TechParams, alpha, m_elast):
    """Characteristic coefficients on the (alpha, vartheta*nu) plane.

    The two reduced-form elasticities are the only technology parameters the
    Jacobian depends on, together with n + delta_K: the saving rate, the
    technology levels and the primary-capacity share cancel at the stationary
    point because iota*y = (n+delta_K)k there.  Along this plane

        i_k = (n+delta_K)(alpha + vartheta nu (1-chi*)/(1-rho sigma) - 1),
        f_k = (n+delta_K) vartheta nu (1-chi*)/(1-rho sigma),

    so raising the material elasticity moves i_k, F_M and G_M together, as
    eqs. (ext_ik_def), (ext_FM_def) and (ext_GM_def) require, while raising the
    capital elasticity moves i_k alone.  tau_M and Pi_M belong to the material
    block and do not move at all.
    """
    d = decomposition(m, t)
    nd = m.n + t.delta_K
    om = (1 - chi_g(m, 0.0)) / (1 - m.rho * m.sigma)
    PhiF, PhiG = d["F_M"] / d["f_k"], d["G_M"] / d["f_k"]
    alpha = np.asarray(alpha, float)
    m_elast = np.asarray(m_elast, float)
    i_k = nd * (alpha + m_elast * om - 1.0)
    f_k = nd * m_elast * om
    a1 = d["tau_M"] - i_k
    a2 = d["Pi_M"] - i_k * d["tau_M"] - f_k * PhiF
    a3 = -i_k * d["Pi_M"] - f_k * PhiG
    return dict(i_k=i_k, f_k=f_k, a1=a1, a2=a2, a3=a3, RH4=a1 * a2 - a3,
                omega=om, tau_M=d["tau_M"], Pi_M=d["Pi_M"],
                PhiF=PhiF, PhiG=PhiG, nd=nd)


def returns_boundaries(m: MaterialParams, t: TechParams) -> dict:
    """Boundary lines in the (alpha, vartheta*nu) plane, as alpha(vartheta*nu).

    a_3 = 0 reduces to the constant-returns line alpha + vartheta*nu = 1 by the
    Schur identity (ext_a3_schur), independently of the material calibration.
    """
    d = decomposition(m, t)
    nd = m.n + t.delta_K
    om = (1 - chi_g(m, 0.0)) / (1 - m.rho * m.sigma)
    PhiF, PhiG = d["F_M"] / d["f_k"], d["G_M"] / d["f_k"]
    return dict(
        a3=lambda M: 1.0 - M * om * (1.0 + PhiG / d["Pi_M"]),
        a1=lambda M: 1.0 + d["tau_M"] / nd - M * om,
        a2=lambda M: 1.0 + (d["Pi_M"] - nd * M * om * PhiF)
        / (nd * d["tau_M"]) - M * om,
    )


def max_abs_imag(m: MaterialParams, t: TechParams, alphas, m_elasts) -> float:
    """Largest |Im| of any characteristic root over a grid of the plane."""
    worst = 0.0
    for A in np.atleast_1d(alphas):
        for M in np.atleast_1d(m_elasts):
            c = returns_coeffs(m, t, A, M)
            ev = np.roots([1.0, float(c["a1"]), float(c["a2"]), float(c["a3"])])
            worst = max(worst, float(np.abs(ev.imag).max()))
    return worst


def root_separation(m: MaterialParams, t: TechParams) -> dict:
    """The separating rate gbar = G_M/F_M and the material decay rates.

    Writing phi = f_k (c_U s_u + c_D sigma s_d) for the coupling strength and
    gbar = G_M/F_M, the characteristic polynomial factorises as

        f(zeta) = (zeta - i_k)(zeta + mu_1)(zeta + mu_2) - phi (zeta + gbar),

    where -mu_1, -mu_2 are the (always real) eigenvalues of the Metzler
    material block.  The perturbation vanishes at zeta = -gbar, so that point
    is fixed as the coupling varies.  If gbar lies strictly between mu_1 and
    mu_2 the pole of h(zeta) = (zeta - i_k)(zeta + mu_1)(zeta + mu_2)/(zeta +
    gbar) separates the branches on which h runs from 0 to +infinity, and
    h(zeta) = phi has three real solutions for every phi > 0: no complex pair
    can form at any investment behaviour or any feedback strength.  Oscillatory
    adjustment therefore requires gbar outside (mu_1, mu_2).
    """
    d = decomposition(m, t)
    mu = np.sort(-np.roots([1.0, d["tau_M"], d["Pi_M"]]).real)
    gbar = d["G_M"] / d["F_M"]
    return dict(mu1=float(mu[0]), mu2=float(mu[1]), gbar=float(gbar),
                separated=bool(mu[0] < gbar < mu[1]),
                phi_unit=d["F_M"] / d["f_k"], gbar_unit=d["G_M"] / d["f_k"],
                tau_M=d["tau_M"], Pi_M=d["Pi_M"], i_k=d["i_k"], f_k=d["f_k"])


def oscillation_boundary(m: MaterialParams, t: TechParams, i_k):
    """Exact boundary of the oscillatory region in the coupling strength.

    A complex pair appears exactly when f has a double root, and by
    eq. (ext_charpoly_factorised) that happens precisely at the critical points
    of the rational function R of eq. (ext_root_ratio): f = 0 gives
    phi = R(zeta) and f' = 0 gives phi = P'(zeta), and the two coincide iff
    P(zeta) = (zeta + g)P'(zeta), which is R'(zeta) = 0.

    Returns, for each i_k, the sorted positive critical values of R.  There are
    either none --- in which case the roots are real for every coupling
    strength --- or two, and the cubic has a complex pair exactly for phi
    strictly between them.  The construction is closed form: the critical
    points solve a cubic in zeta.
    """
    r = root_separation(m, t)
    mu1, mu2, g = r["mu1"], r["mu2"], r["gbar"]
    out = []
    for ik in np.atleast_1d(np.asarray(i_k, float)):
        P = np.poly([ik, -mu1, -mu2])              # descending coefficients
        Q = np.polysub(P, np.polymul([1.0, g], np.polyder(P)))
        zc = np.roots(Q)
        zc = zc[np.abs(zc.imag) < 1e-11].real
        vals = sorted(float(np.polyval(P, z) / (z + g))
                      for z in zc if abs(z + g) > 1e-14
                      and np.polyval(P, z) / (z + g) > 0)
        out.append(vals)
    return out


def coeffs_from_capital(m: MaterialParams, t: TechParams, i_k, f_k):
    """Characteristic coefficients as functions of the two capital-side objects.

    tau_M and Pi_M belong to the material block and are held fixed; F_M and
    G_M are both proportional to f_k, so (i_k, f_k) spans every configuration
    a generalised investment schedule i(y,k) with non-negative marginal
    propensity can produce at a given material calibration.
    """
    d = decomposition(m, t)
    PhiF, PhiG = d["F_M"] / d["f_k"], d["G_M"] / d["f_k"]
    i_k, f_k = np.asarray(i_k, float), np.asarray(f_k, float)
    return (d["tau_M"] - i_k,
            d["Pi_M"] - i_k * d["tau_M"] - f_k * PhiF,
            -i_k * d["Pi_M"] - f_k * PhiG)


def material_block_bounds(m: MaterialParams, t: TechParams) -> dict:
    """Structural restrictions on the material aggregates (tau_M, Pi_M).

    M is a 2x2 Metzler matrix, so its eigenvalues are real and

        tau_M^2 >= 4 Pi_M,

    which bounds tau_M below by 2 sqrt(Pi_M).  Together with tau_M >= 2n this
    is what keeps the model out of the oscillatory corner of the (i_k, tau_M)
    plane: the plane holds Pi_M fixed, whereas in the model tau_M cannot be
    reduced without reducing Pi_M by the square of the same factor, since both
    vanish together as chi* -> 1.
    """
    d = decomposition(m, t)
    return dict(tau_M=d["tau_M"], Pi_M=d["Pi_M"],
                tau_min=2 * np.sqrt(d["Pi_M"]),
                discriminant=d["tau_M"] ** 2 - 4 * d["Pi_M"],
                material_eigenvalues=np.linalg.eigvals(jacobian3(m, t)[1:, 1:]))


def tau_M_decomposition(m: MaterialParams) -> dict:
    """tau_M as a sum of five non-negative terms.

        tau_M = 2n + lam(1-kappa) + lam kappa (1-rho)/(1-rho sigma)
                   + gam(1-q)     + q gam (1-sigma)/(1-rho sigma).

    The identity is exact, so on the admissible parameter set --- n > 0,
    kappa, q, rho, sigma in [0,1], lam, gam >= 0 and 1 - rho sigma > 0 --- one
    has tau_M >= 2n > 0 with no appeal to any stability condition.  Term by
    term, each recovery feedback is capped by its own turnover rate:

        c_U s_u       = kappa lam (1-sigma) rho / (1-rho sigma) <= kappa lam <= lam,
        c_D sigma s_d = q gam (1-rho) sigma  / (1-rho sigma)    <= q gam     <= gam,

    since 1 - (1-sigma)rho/(1-rho sigma) = (1-rho)/(1-rho sigma) >= 0 and
    1 - (1-rho)sigma/(1-rho sigma) = (1-sigma)/(1-rho sigma) >= 0.

    NOTE.  This is a property of the material block of the growth extension, in
    which primary supply is produced by capital and psi does not appear in
    eq. (ext_s_reduced).  In the baseline model of Section 2 the denominator is
    1 - psi - rho sigma, both bounds fail once psi > 0, and the trace condition
    is a genuine restriction --- one implied by Gamma < Gamma_M rather than by
    the parameter ranges.
    """
    den = 1 - m.rho * m.sigma
    terms = dict(
        dilution=2 * m.n,
        durability_loss=m.lam * (1 - m.kappa),
        uncollected_eol=m.lam * m.kappa * (1 - m.rho) / den,
        unrecoverable_discard=m.gamma * (1 - m.q),
        fabricated_out=m.q * m.gamma * (1 - m.sigma) / den,
    )
    tau = ((m.n + m.lam) + (m.n + m.gamma)
           - (1 - m.sigma) * m.kappa * m.rho * m.lam / den
           - m.q * (1 - m.rho) * m.sigma * m.gamma / den)
    return dict(tau_M=tau, terms=terms, total=sum(terms.values()),
                lower_bound=2 * m.n)


def hopf_period(a2: float) -> float:
    """Linearised period at the Hopf boundary, eq. (ext_hopf_period)."""
    return 2 * np.pi / np.sqrt(a2) if a2 > 0 else np.nan


def complex_pair(a1: float, a2: float, a3: float) -> bool:
    """Depressed-cubic discriminant test, eq. (ext_complex_pair_condition)."""
    pc = a2 - a1 ** 2 / 3
    qc = 2 * a1 ** 3 / 27 - a1 * a2 / 3 + a3
    return bool(qc ** 2 / 4 + pc ** 3 / 27 > 0)


# ----------------------------------------------------------------------
# Balanced growth on the knife edge alpha + vartheta*nu = 1
# ----------------------------------------------------------------------

# The balanced-growth knife edge alpha + vartheta*nu = 1 is only compatible with
# the constant-returns normalisation alpha + nu + varsigma = 1 and varsigma >= 0
# at vartheta = 1 (whence varsigma = 0): otherwise varsigma = -(1-vartheta)nu < 0.
# The illustration below therefore uses the AK-like configuration vartheta = 1,
# alpha = 1 - nu, in which g_s = g_k.
KNIFE = TechParams(alpha=0.95, nu=0.05, theta_v=1.0, delta_K=0.05,
                   iota=0.107, A_Y=1.0, B=0.5, b_P=0.16)


def knife_edge(t: TechParams = KNIFE) -> TechParams:
    """Set alpha so that alpha + vartheta*nu = 1 exactly."""
    return replace(t, alpha=1.0 - t.theta_v * t.nu)


def g_map(g_k: float, m: MaterialParams, t: TechParams) -> float:
    """Right-hand side of eq. (ext_gk_equation) minus g_k."""
    chi = chi_g(m, t.theta_v * g_k)
    if chi >= 1:
        return np.inf
    return (t.iota * t.A_Y * (1 - t.b_P) ** t.alpha
            * ((1 - m.sigma) * t.B * t.b_P ** t.theta_v / (1 - chi)) ** t.nu
            - (m.n + t.delta_K) - g_k)


def balanced_growth_rate(m: MaterialParams, t: TechParams,
                         lo=-0.05, hi=2.0) -> float:
    """Unique balanced growth rate, Proposition (app_gk_unique)."""
    return brentq(g_map, lo, hi, args=(m, t), xtol=1e-14, rtol=1e-15)


def detrended_jacobian(m: MaterialParams, t: TechParams, g_k: float) -> np.ndarray:
    """Planar Jacobian of the detrended ratios on the knife edge, eq. (app_Jtilde)."""
    g_s = t.theta_v * g_k
    den = 1 - m.rho * m.sigma
    s_u, s_d = m.rho * m.lam / den, m.gamma / den
    cU, cD = (1 - m.sigma) * m.kappa, m.q * (1 - m.rho)
    s_hat = t.B * t.b_P ** t.theta_v / (1 - chi_g(m, g_s))
    u_hat, d_hat = BU(m, g_s) * s_hat, BD(m, g_s) * s_hat
    M = np.array([[cU * s_u - (m.n + m.lam + g_s), cU * s_d],
                  [cD * (m.lam + m.sigma * s_u),
                   cD * m.sigma * s_d - (m.n + m.gamma + g_s)]])
    a_g = t.theta_v * t.nu * (g_k + m.n + t.delta_K) / s_hat
    return M - a_g * np.outer([u_hat, d_hat], [s_u, s_d])


# ----------------------------------------------------------------------
# Comparative statics
# ----------------------------------------------------------------------

def chi_derivatives(m: MaterialParams, g: float = 0.0) -> dict:
    """chi* derivatives, eqs. (ext_chi_rho_q)-(ext_chi_kappa_gamma)."""
    bu, bd = BU(m, g), BD(m, g)
    nl, ng = m.n + m.lam + g, m.n + m.gamma + g
    recov = m.rho + m.gamma * m.q * (1 - m.rho) / ng
    return dict(
        rho=(m.sigma + m.lam * bu) * (ng - m.gamma * m.q) / ng,
        q=m.gamma * bd / m.q if m.q > 0 else np.nan,
        kappa=m.lam * (1 - m.sigma) / nl * recov,
        gamma=(m.n + g) / ng * bd,
        sigma=(m.n + g + m.lam * (1 - m.kappa)) / nl * recov,
    )


def level_elasticities(m: MaterialParams, t: TechParams) -> dict:
    """d ln k*/d chi*, d ln s*/d chi* and the parameter elasticities."""
    chi = chi_g(m, 0.0)
    dk = t.nu / ((1 - t.returns) * (1 - chi))
    ds = (1 - t.alpha) / ((1 - t.returns) * (1 - chi))
    cd = chi_derivatives(m)
    return dict(dlnk_dchi=dk, dlns_dchi=ds,
                dlnk={z: dk * v for z, v in cd.items()},
                dlns={z: ds * v for z, v in cd.items()})


def dgk_dz(m: MaterialParams, t: TechParams, g_k: float, z: str) -> float:
    """Growth response on the knife edge, eq. (ext_dgk_dz)."""
    g_s = t.theta_v * g_k
    chi = chi_g(m, g_s)
    chi_z = chi_derivatives(m, g_s)[z]
    return (t.nu * (g_k + m.n + t.delta_K) * chi_z
            / (1 - chi + t.nu * t.theta_v * (g_k + m.n + t.delta_K)
               * abs(dchi_dg(m, g_s))))
