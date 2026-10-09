"""
mm_optimal.py
=============
The optimal-growth block of Chapter 2: the intensity-sustainable planner problem
of Section 6.6, its detrended canonical state-costate system, the cubic
criterion of Appendix B and the two extensions (imperfect substitutability pi,
endogenous recovery capacity rho).

Detrending
----------
All per-capita quantities grow at g_I on the balanced path, and the costates
decline at theta_C * g_I.  Writing z = (k, u, d) for the detrended states and
w = (w_K, w_U, w_D) for the detrended costates, and folding the dilution term
g_I into the Hamiltonian,

    Hhat = W(c,x,u,d)
         + w_K [ y - c - (n + delta_K + g_I) k ]
         + w_U [ (1-sigma) kappa s - (n + lambda + g_I) u ]
         + w_D [ q(1-rho)(lambda u + sigma s) - (n + gamma + g_I) d ],

the canonical system is exactly

    zdot = Hhat_w,        wdot = rho_e * w - Hhat_z,
    rho_e = rho_W - (1 - theta_C) g_I,

so that tr(J6) = 3 rho_e and the six eigenvalues pair with sum rho_e
(eq. oc_J6_blocks, oc_effective_discount).

Welfare marginal valuations are iso-elastic with common curvature theta_C:

    W_c = c^-theta,  W_x = w_X x^-theta,  W_u = w_U u^-theta,  W_d = -w_D d^-theta,

which is what keeps the shadow ratios constant along the path
(Proposition oc_intensity_growth).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import root

__all__ = ["OCParams", "COPPER_OC", "StationaryPoint",
           "controls", "vector_field", "jacobian6", "invariants",
           "solve_stationary", "dockner_ok", "balanced_ratios",
           "rho_hill", "solve_stationary_rho", "branch_scan", "find_fold",
           "inflection", "allocation_hessian"]


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class OCParams:
    # --- material block (copper calibration with urban mining) -----------
    lam: float = 0.0262
    sigma: float = 0.170
    kappa: float = 0.994
    q: float = 0.630
    gamma: float = 0.002
    rho: float = 0.588
    n: float = 0.01

    # --- technology and preferences (illustrative, not calibrated) -------
    alpha: float = 0.35
    nu: float = 0.05
    theta_v: float = 0.6        # vartheta, primary-capacity capital elasticity
    delta_K: float = 0.05
    theta_C: float = 2.0
    rho_W: float = 0.03
    AYAM: float = 1.0           # A_Y0 * A_M0^nu
    APB: float = 0.5            # A_P0 * B

    # --- growth side -----------------------------------------------------
    g_I: float = 0.015
    g_M: float = 0.01
    eta: float = 1.0
    delta: float = 0.020

    # --- welfare weights on services, in-use stock, discard --------------
    w_X: float = 0.02
    w_U: float = 0.01
    w_D: float = 0.02

    # --- substitutability (Appendix pi-extension) ------------------------
    pi: float = 1.0

    # --- endogenous recovery (Appendix rho-extension); None = inactive ---
    rho_max: float | None = None
    h: float = 2.0
    theta_R: float = 0.6
    m_bar: float = 0.08

    # derived -------------------------------------------------------------
    @property
    def varsigma(self) -> float:
        return 1.0 - self.alpha - self.nu

    @property
    def g_A(self) -> float:
        return (1 - self.alpha - self.nu) * self.g_I - self.nu * self.g_M

    @property
    def g_P(self) -> float:
        return self.eta * self.delta + (1 - self.theta_v) * self.g_I

    @property
    def rho_e(self) -> float:
        return self.rho_W - (1 - self.theta_C) * self.g_I

    @property
    def c_U(self) -> float:
        return (1 - self.sigma) * self.kappa

    @property
    def c_D(self) -> float:
        return self.q * (1 - self.rho)


COPPER_OC = OCParams()


# ----------------------------------------------------------------------
# Growth-adjusted stock-flow ratios and circularity
# ----------------------------------------------------------------------

def balanced_ratios(p: OCParams, g: float | None = None, rho: float | None = None):
    """B_U(g), B_D(g) and chi_I(g), eqs. (oc_BU_BD), (oc_chiI)."""
    g = p.g_I if g is None else g
    rho = p.rho if rho is None else rho
    BU = (1 - p.sigma) * p.kappa / (p.n + p.lam + g)
    BD = p.q * (1 - rho) * (p.lam * BU + p.sigma) / (p.n + p.gamma + g)
    chi = rho * p.sigma + rho * p.lam * BU + p.gamma * BD
    return BU, BD, chi


def rho_hill(m: float, p: OCParams) -> tuple[float, float, float]:
    """Hill recovery technology rho(m) and its first two derivatives."""
    if p.rho_max is None:
        return p.rho, 0.0, 0.0
    h, mb, rmax = p.h, p.m_bar, p.rho_max
    m = max(m, 1e-14)
    A, Bd = m ** h, mb ** h
    den = Bd + A
    r = rmax * A / den
    dr = rmax * h * Bd * m ** (h - 1) / den ** 2
    d2r = (rmax * h * Bd * m ** (h - 2)
           * ((h - 1) * den - (h + 1) * A) / den ** 3)
    return r, dr, d2r


# ----------------------------------------------------------------------
# Hamiltonian, controls, vector field
# ----------------------------------------------------------------------

def _welfare_marginals(c, x, u, d, p: OCParams):
    th = p.theta_C
    return (c ** -th, p.w_X * x ** -th, p.w_U * u ** -th, -p.w_D * d ** -th)


def _welfare(c, x, u, d, p: OCParams):
    th = p.theta_C

    def crra(z, weight=1.0):
        return weight * (z ** (1 - th) - 1) / (1 - th)

    return crra(c) + crra(x, p.w_X) + crra(u, p.w_U) - crra(d, p.w_D)


def _block(z, w, ctrl, p: OCParams):
    """Everything downstream of the states and controls, in one place."""
    k, u, d = z
    wK, wU, wD = w
    if p.rho_max is None:
        bP, lP = ctrl
        bR = lR = 0.0
        mR, rho, drho = 0.0, p.rho, 0.0
    else:
        bP, lP, bR, lR = ctrl
        mR = (bR * k) ** p.theta_R * lR ** (1 - p.theta_R)
        rho, drho, _ = rho_hill(mR, p)

    # clamp to the interior: the root solver probes shares outside (0,1)
    eps = 1e-12
    lP, lR, bP, bR = (max(v, eps) for v in (lP, lR, bP, bR))
    lY = max(1.0 - lP - lR, eps)
    bY = max(1.0 - bP - bR, eps)
    k = max(k, eps)
    pp = p.APB * lP ** (1 - p.theta_v) * (bP * k) ** p.theta_v
    den = 1.0 - rho * p.sigma
    s = (pp + rho * p.lam * u + p.gamma * d) / den
    x = (1 - p.sigma) * (p.pi * s + (1 - p.pi) * pp)
    y = p.AYAM * lY ** p.varsigma * (bY * k) ** p.alpha * x ** p.nu
    c = wK ** (-1.0 / p.theta_C)
    return dict(k=k, u=u, d=d, wK=wK, wU=wU, wD=wD, bP=bP, lP=lP, bR=bR, lR=lR,
                lY=lY, bY=bY, p=pp, s=s, x=x, y=y, c=c, rho=rho, drho=drho,
                mR=mR, den=den)


def _hamiltonian(z, w, ctrl, p: OCParams) -> float:
    B = _block(z, w, ctrl, p)
    cU = (1 - p.sigma) * p.kappa
    cD = p.q * (1 - B["rho"])
    W = _welfare(B["c"], B["x"], B["u"], B["d"], p)
    return (W
            + B["wK"] * (B["y"] - B["c"] - (p.n + p.delta_K + p.g_I) * B["k"])
            + B["wU"] * (cU * B["s"] - (p.n + p.lam + p.g_I) * B["u"])
            + B["wD"] * (cD * (p.lam * B["u"] + p.sigma * B["s"])
                         - (p.n + p.gamma + p.g_I) * B["d"]))


def _margin_factory(z, w, ctrl, p: OCParams):
    """Return dH(dp, drho, dlny_direct): the marginal Hamiltonian of a
    perturbation that changes primary supply by dp, the recovery rate by drho
    and final output directly (through the factor left in the final sector or
    through capital itself) by a proportional dlny_direct."""
    B = _block(z, w, ctrl, p)
    k, u, d = z
    wK, wU, wD = w
    rho, den, pi = B["rho"], B["den"], p.pi
    cU = (1 - p.sigma) * p.kappa
    _, Wx, _, _ = _welfare_marginals(B["c"], B["x"], u, d, p)
    y, x, s = B["y"], B["x"], B["s"]
    collectable = p.lam * u + p.sigma * s          # = r/rho
    s_ro = collectable / den                       # eq. (rho_s_rho)

    def dH(dp=0.0, drho=0.0, dlny_direct=0.0):
        ds = dp / den + drho * s_ro
        dx = (1 - p.sigma) * (pi * ds + (1 - pi) * dp)
        dy = y * (dlny_direct + p.nu * dx / x)
        return (Wx * dx + wK * dy + wU * cU * ds
                + wD * (p.q * (1 - rho) * p.sigma * ds - p.q * drho * collectable))

    return B, dH


def _foc(ctrl, z, w, p: OCParams):
    """Allocation first-order conditions of the maximised Hamiltonian.

    Capital and labour margins for the primary sector, eqs. (oc_bP_foc) and
    (oc_lP_foc); when endogenous recovery is active, also the two recycling
    margins of eqs. (rho_three_way_foc)-(rho_labour_foc).
    """
    B, dH = _margin_factory(z, w, ctrl, p)
    pp = B["p"]
    fY = -p.alpha / B["bY"]          # d ln y from capital taken out of final use
    fYl = -p.varsigma / B["lY"]      # same for labour
    out = [dH(dp=p.theta_v * pp / B["bP"], dlny_direct=fY),
           dH(dp=(1 - p.theta_v) * pp / B["lP"], dlny_direct=fYl)]
    if p.rho_max is not None:
        mR, drho_dm = B["mR"], B["drho"]
        out.append(dH(drho=drho_dm * p.theta_R * mR / B["bR"], dlny_direct=fY))
        out.append(dH(drho=drho_dm * (1 - p.theta_R) * mR / B["lR"], dlny_direct=fYl))
    return np.array(out)


def _logit(v):
    v = np.clip(np.asarray(v, float), 1e-12, 1 - 1e-12)
    return np.log(v / (1 - v))


def _expit(t):
    return 1.0 / (1.0 + np.exp(-np.asarray(t, float)))


def controls(z, w, p: OCParams, guess=None):
    """Optimal allocation shares given states and costates.

    Solved in logit coordinates so the shares stay strictly interior; this
    keeps the control map smooth, which the numerical Jacobian relies on.
    """
    if guess is None:
        guess = [0.16, 0.07] if p.rho_max is None else [0.15, 0.065, 0.06, 0.026]
    sol = root(lambda t: _foc(_expit(t), z, w, p), _logit(guess),
               method="hybr", tol=1e-14)
    return _expit(sol.x)


def _dH_dz(z, w, ctrl, p: OCParams):
    """Partial derivatives of Hhat with respect to the states, controls fixed.

    When recovery capacity is endogenous the bundle depends on the state
    through k, so the recovery rate carries a channel of its own,
    d rho / d k = rho'(m_R) theta_R m_R / k, which enters H_k alongside the
    primary-capacity channel.  It vanishes at the linear corner, where
    rho'(0) = 0 for h > 1.
    """
    B, dH = _margin_factory(z, w, ctrl, p)
    k, u, d = z
    wK, wU, wD = w
    rho, den = B["rho"], B["den"]
    cU = (1 - p.sigma) * p.kappa
    cD = p.q * (1 - rho)
    _, Wx, Wu, Wd = _welfare_marginals(B["c"], B["x"], u, d, p)

    # --- capital -------------------------------------------------------
    drho_dk = B["drho"] * p.theta_R * B["mR"] / k if p.rho_max is not None else 0.0
    Hk = dH(dp=p.theta_v * B["p"] / k, drho=drho_dk,
            dlny_direct=p.alpha / k) - wK * (p.n + p.delta_K + p.g_I)

    # --- material stocks (rho does not depend on u or d) ---------------
    s_u = rho * p.lam / den
    s_d = p.gamma / den
    x_u = (1 - p.sigma) * p.pi * s_u
    x_d = (1 - p.sigma) * p.pi * s_d
    y_u = B["y"] * p.nu * x_u / B["x"]
    y_d = B["y"] * p.nu * x_d / B["x"]
    Hu = (Wx * x_u + Wu + wK * y_u
          + wU * (cU * s_u - (p.n + p.lam + p.g_I))
          + wD * cD * (p.lam + p.sigma * s_u))
    Hd = (Wx * x_d + Wd + wK * y_d + wU * cU * s_d
          + wD * (cD * p.sigma * s_d - (p.n + p.gamma + p.g_I)))
    return np.array([Hk, Hu, Hd])


def vector_field(y6, p: OCParams, guess=None):
    """The detrended canonical vector field F(z, w) in R^6."""
    z, w = np.asarray(y6[:3], float), np.asarray(y6[3:], float)
    ctrl = controls(z, w, p, guess)
    B = _block(z, w, ctrl, p)
    cU = (1 - p.sigma) * p.kappa
    cD = p.q * (1 - B["rho"])
    zdot = np.array([
        B["y"] - B["c"] - (p.n + p.delta_K + p.g_I) * B["k"],
        cU * B["s"] - (p.n + p.lam + p.g_I) * B["u"],
        cD * (p.lam * B["u"] + p.sigma * B["s"]) - (p.n + p.gamma + p.g_I) * B["d"],
    ])
    wdot = p.rho_e * w - _dH_dz(z, w, ctrl, p)
    return np.concatenate([zdot, wdot])


# ----------------------------------------------------------------------
# Stationary point, Jacobian and invariants
# ----------------------------------------------------------------------

@dataclass
class StationaryPoint:
    p: OCParams
    z: np.ndarray
    w: np.ndarray
    ctrl: np.ndarray
    J6: np.ndarray
    K: tuple
    omega: np.ndarray
    eigenvalues: np.ndarray
    residual: float

    # readable summary -------------------------------------------------
    def summary(self) -> dict:
        B = _block(self.z, self.w, self.ctrl, self.p)
        y = B["y"]
        out = dict(
            bP=B["bP"], lP=B["lP"], c_over_y=B["c"] / y, k_over_y=B["k"] / y,
            I_S=B["s"] / y, I_U=B["u"] / y, I_D=B["d"] / y, I_P=B["p"] / y,
            rho=B["rho"], y=y,
        )
        if self.p.rho_max is not None:
            out.update(bR=B["bR"], lR=B["lR"], mR=B["mR"])
        return out

    def half_life(self) -> float:
        stable = [e.real for e in self.eigenvalues if e.real < 0]
        return np.log(2) / abs(max(stable))


def jacobian6(y6, p: OCParams, guess=None, step=1e-6) -> np.ndarray:
    """Numerical Jacobian of the canonical vector field (central differences)."""
    y6 = np.asarray(y6, float)
    if guess is None:
        guess = controls(y6[:3], y6[3:], p)
    J = np.empty((6, 6))
    for j in range(6):
        hstep = step * max(abs(y6[j]), 1e-2)
        f = []
        for m in (-2, -1, 1, 2):
            v = y6.copy()
            v[j] += m * hstep
            f.append(vector_field(v, p, guess))
        # five-point stencil (fourth-order accurate)
        J[:, j] = (f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * hstep)
    return J


def _principal_minor_sums(J: np.ndarray) -> list[float]:
    """c_m = sum of the m x m principal minors of J, m = 1..6."""
    from itertools import combinations
    n = J.shape[0]
    cs = []
    for m in range(1, n + 1):
        tot = 0.0
        for idx in combinations(range(n), m):
            ix = np.ix_(idx, idx)
            tot += np.linalg.det(J[ix])
        cs.append(tot)
    return cs


def invariants(J: np.ndarray, rho_e: float):
    """K_1, K_2, K_3 and the omega roots, eq. (oc_omega_cubic)."""
    c = _principal_minor_sums(J)
    K1 = c[1] - 3 * rho_e ** 2
    K2 = c[3] - rho_e ** 2 * K1
    K3 = c[5]
    omega = np.roots([1.0, -K1, K2, -K3])
    checks = dict(c1=c[0], c1_target=3 * rho_e,
                  c3=c[2], c3_target=2 * rho_e * c[1] - 5 * rho_e ** 3,
                  c5=c[4], c5_target=rho_e * K2)
    return (K1, K2, K3), omega, checks


def dockner_ok(omega: np.ndarray, rho_e: float) -> bool:
    """Every omega inside D = {Re w < (Im w)^2 / rho_e^2}, eq. (oc_dockner_region)."""
    return bool(np.all(omega.real < omega.imag ** 2 / rho_e ** 2))


def _initial_guess(p: OCParams):
    """Analytic starting point from the balanced-path algebra."""
    BU, BD, chi = balanced_ratios(p)
    RK = p.theta_C * p.g_I + p.rho_W
    kY_y = p.alpha / (RK + p.n + p.delta_K)
    bP = 0.16
    IK = kY_y / (1 - bP)
    lP = 0.07
    # solve the two identities for y and I_S
    for _ in range(200):
        IS = ((1 - bP) ** -p.alpha * (1 - lP) ** -p.varsigma) ** 0.0 + 0.0  # placeholder
        break
    # iterate: I_P = (1-chi) I_S with both identities
    y = 1.5
    for _ in range(500):
        IS = (y ** (1 - p.alpha - p.nu)
              / ((1 - lP) ** p.varsigma * (1 - bP) ** p.alpha
                 * (1 - p.sigma) ** p.nu * IK ** p.alpha)) ** (1 / p.nu)
        IP = p.APB * lP ** (1 - p.theta_v) * bP ** p.theta_v * IK ** p.theta_v * y ** (p.theta_v - 1)
        f = IP - (1 - chi) * IS
        y *= (1 + 0.02 * np.sign(f))
        if abs(f) < 1e-12:
            break
    s = IS * y
    k = IK * y
    u, d = BU * s, BD * s
    c = (1 - (p.n + p.delta_K + p.g_I) * IK) * y
    wK = c ** -p.theta_C
    return np.array([k, u, d, wK, 0.15 * wK, -0.15 * wK])


def solve_stationary(p: OCParams = COPPER_OC, guess=None,
                     ctrl_guess=None) -> StationaryPoint:
    """Solve F(z, w) = 0 and characterise the stationary point."""
    y0 = _initial_guess(p) if guess is None else np.asarray(guess, float)
    sol = root(lambda v: vector_field(v, p, ctrl_guess), y0,
               method="hybr", tol=1e-14)
    if not sol.success:
        sol = root(lambda v: vector_field(v, p, ctrl_guess), sol.x,
                   method="lm", tol=1e-14)
    y6 = sol.x
    ctrl = controls(y6[:3], y6[3:], p, ctrl_guess)
    J = jacobian6(y6, p, ctrl_guess)
    K, omega, checks = invariants(J, p.rho_e)
    ev = np.linalg.eigvals(J)
    return StationaryPoint(
        p=p, z=y6[:3], w=y6[3:], ctrl=ctrl, J6=J, K=K,
        omega=np.sort_complex(omega), eigenvalues=np.sort_complex(ev),
        residual=float(np.max(np.abs(vector_field(y6, p, ctrl_guess)))),
    )


# ----------------------------------------------------------------------
# Endogenous recovery: the two configurations and the saddle-node scan
# ----------------------------------------------------------------------

def allocation_hessian(sp: "StationaryPoint", step: float = 1e-5) -> np.ndarray:
    """Hessian of Hhat in the allocation shares at a stationary point.

    Negative definiteness is the allocation second-order condition; it fails on
    the locally-increasing-returns branch m_R < m_R^infl of the Hill technology.
    """
    n = len(sp.ctrl)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hi = step * max(abs(sp.ctrl[i]), 1e-3)
            hj = step * max(abs(sp.ctrl[j]), 1e-3)
            acc = 0.0
            for si, sj, w in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
                c = sp.ctrl.copy()
                c[i] += si * hi
                c[j] += sj * hj
                acc += w * _hamiltonian(sp.z, sp.w, c, sp.p)
            H[i, j] = acc / (4 * hi * hj)
    return 0.5 * (H + H.T)


# Control-side starting points that select each branch of the recycling margin:
# the saturated branch (large bundle) and the increasing-returns branch (small
# bundle) are distinct roots of the same first-order condition.
_CTRL_CIRCULAR = [0.145, 0.065, 0.059, 0.026]
_CTRL_MIDDLE = [0.190, 0.085, 0.004, 0.0018]


def solve_stationary_rho(p: OCParams, branch: str = "circular",
                         guess=None, ctrl_guess=None) -> StationaryPoint:
    """Stationary configuration of the endogenous-recovery system.

    branch='linear'  : the corner b_R = l_R = 0, rho = 0.  Since rho'(0) = 0 for
                       h > 1 this satisfies the recycling margins identically,
                       and the canonical system coincides with the maintained
                       model at rho = 0.
    branch='circular': the interior configuration on the saturated branch
                       m_R > m_R^infl.
    branch='middle'  : the interior configuration on the locally-increasing-
                       returns branch m_R < m_R^infl -- the Skiba point at which
                       the allocation second-order condition degenerates.
    """
    if branch == "linear":
        return solve_stationary(replace(p, rho_max=None, rho=0.0), guess=guess)
    cg = (_CTRL_CIRCULAR if branch == "circular" else _CTRL_MIDDLE) \
        if ctrl_guess is None else ctrl_guess
    return solve_stationary(p, guess=guess, ctrl_guess=cg)


def inflection(p: OCParams) -> float:
    """Scale at which the Hill technology changes convexity, eq. (rho_inflection)."""
    return p.m_bar * ((p.h - 1) / (p.h + 1)) ** (1 / p.h)


def branch_scan(p: OCParams, m_bars, branch: str = "circular",
                guess=None, tol=1e-10, jump=0.03):
    """Continuation of one interior branch in the half-saturation scale.

    Returns rows (m_bar, rho*, m_R*, K_3).  The continuation stops where the
    branch can no longer be followed: for the circular branch that is the
    saddle-node at which it merges with the middle configuration and both
    disappear, leaving the linear economy as the only long-run configuration.
    """
    rows, g, ctrl, last = [], guess, None, None
    saturated = branch == "circular"
    for mb in m_bars:
        q = replace(p, m_bar=mb)
        try:
            sp = solve_stationary_rho(q, branch, guess=g, ctrl_guess=ctrl)
        except Exception:
            break
        if sp.residual > tol:
            break
        S = sp.summary()
        infl = inflection(q)
        on_branch = (S["mR"] > infl * 1.001) if saturated else (S["mR"] < infl * 0.999)
        if not on_branch:
            break
        if last is not None and abs(S["rho"] - last) > jump:
            break
        g, ctrl, last = np.concatenate([sp.z, sp.w]), sp.ctrl, S["rho"]
        rows.append((mb, S["rho"], S["mR"], sp.K[2]))
    return rows


# backwards-compatible alias
def saddle_node_scan(p: OCParams, m_bars, guess=None, tol=1e-10, jump=0.03):
    return branch_scan(p, m_bars, "circular", guess=guess, tol=tol, jump=jump)


def find_fold(p: OCParams, lo: float, hi: float, guess=None, iters: int = 30,
              start: float | None = None):
    """Bisect on the half-saturation scale to locate the saddle-node.

    Each trial continues the circular branch from `start` (the parameterisation
    at which the branch is known to exist) up to the trial scale.
    """
    grid_lo = p.m_bar if start is None else start
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        grid = np.append(np.arange(grid_lo, mid, 5e-4), mid)
        rows = branch_scan(p, grid, "circular", guess=guess)
        if rows and abs(rows[-1][0] - mid) < 1e-12:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5:
            break
    return lo, hi
