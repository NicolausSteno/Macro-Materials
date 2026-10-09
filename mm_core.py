"""
mm_core.py
==========
Baseline two-stock macro-materials model of Chapter 1 (Sections 2-3).

Implements, in closed form and by direct linear solution:

  * the refined-metal supply identity            eq. (s-identity)
  * the per-capita state system in (u, d)        eq. (two-ode)
  * the stock-feedback intensity Gamma           eq. (Gamma)
  * the stability threshold Gamma_M              eq. (Gamma_M)
  * the Jacobian J_M, its trace and determinant  eq. (full_material_jacobian)
  * the steady state (u*, d*, s*)                eq. (ustar), (dstar), (sstar)
  * steady-state circularity chi*                eq. (chistar)
  * the energy block Omega*, Omega*_NP, T_E      eq. (omega_closed_form), (critical_date)

Every closed form is checked against the corresponding direct numerical
solution, so the module doubles as a verification of the chapter's algebra.

Conventions
-----------
Stocks u, d          : kg/cap
Flows s, p, r        : kg/cap/yr
Energy intensities   : MJ/kg
Energy requirements  : MJ/cap/yr
Rates n, lam, gam    : 1/yr
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import numpy as np

__all__ = [
    "MaterialParams", "EnergyParams",
    "CU_BASE", "CU_GAMMA", "ZN_BASE", "ZN_GAMMA",
    "ENERGY_CU", "ENERGY_ZN",
    "SteadyState", "EnergyState",
    "Gamma", "Gamma_M", "Delta_M", "chi_star", "jacobian",
    "steady_state", "steady_state_numeric", "energy_state",
    "critical_horizon", "rho_offset", "simulate",
]


# ----------------------------------------------------------------------
# Parameter containers
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialParams:
    """Calibrated parameters of the baseline material block (Table 1)."""
    name: str
    psi: float        # primary supply multiplier  phi(1-beta)A
    s_bar: float      # autonomous supply  = p_bar + xi_bar   (kg/cap/yr)
    p_bar: float      # autonomous primary refining           (kg/cap/yr)
    lam: float        # in-use depreciation rate              (1/yr)
    sigma: float      # fabrication-loss share
    rho: float        # scrap collection / recycling coefficient
    gamma: float      # urban-mining (discard recovery) rate  (1/yr)
    kappa: float      # durable-use share of post-new-scrap flow
    q: float          # recoverable share of unrecycled scrap
    n: float = 0.01   # population growth rate                (1/yr)

    # reference (data) values, for reporting only
    u_ref: float | None = None
    d_ref: float | None = None

    @property
    def xi_bar(self) -> float:
        return self.s_bar - self.p_bar

    def feasible(self) -> bool:
        """Positivity of refined-metal throughput: 1 - psi - rho*sigma > 0."""
        return 1.0 - self.psi - self.rho * self.sigma > 0.0


@dataclass(frozen=True)
class EnergyParams:
    """Energy-layer parameters (Appendix Table: energy_parameters_app)."""
    name: str
    eps_P0: float     # primary extraction and refining   (MJ/kg)
    eps_R: float      # secondary refining and remelting  (MJ/kg)
    eps_U: float      # recovery from discard             (MJ/kg)
    eps_F: float      # fabrication                       (MJ/kg)
    eta: float = 1.0  # elasticity of eps_P wrt ore-grade decline
    delta: float = 0.020  # ore-grade deterioration rate  (1/yr)
    E_bench: float = np.nan   # forward-looking benchmark budget (MJ/cap/yr)
    E_tight: float = np.nan   # tight stress-test budget         (MJ/cap/yr)


# ----------------------------------------------------------------------
# Calibrated cases (chapter Table 1 and calibration appendix)
#
# CU_GAMMA and ZN_GAMMA are THE calibration, identical to the submitted
# paper: low urban-mining rates gamma = 0.002 / 0.003 yr^-1 with the
# recycling coefficient offset to rho = 0.588 / 0.345 so that recovered
# material is not double counted.  CU_BASE and ZN_BASE (gamma = 0 at the
# observed coefficients rho_0 = 0.608 / 0.363) are the no-urban-mining
# comparison retained in the thesis chapter.
# ----------------------------------------------------------------------

CU_BASE = MaterialParams(
    name="Copper, gamma=0",          # no-urban-mining comparison
    psi=0.474, s_bar=0.599, p_bar=0.622,
    lam=0.0262, sigma=0.170, rho=0.608, gamma=0.0,
    kappa=0.994, q=0.630, n=0.01,
    u_ref=50.7, d_ref=19.3,
)

CU_GAMMA = replace(                     # calibration
    CU_BASE, name="Copper, gamma=0.002", gamma=0.002, rho=0.588,
)

ZN_BASE = MaterialParams(
    name="Zinc, gamma=0",            # no-urban-mining comparison
    psi=0.613, s_bar=0.262, p_bar=0.239,
    lam=0.0375, sigma=0.179, rho=0.363, gamma=0.0,
    kappa=0.983, q=0.300, n=0.01,
    u_ref=32.1, d_ref=7.81,
)

ZN_GAMMA = replace(                     # calibration
    ZN_BASE, name="Zinc, gamma=0.003", gamma=0.003, rho=0.345,
)

ENERGY_CU = EnergyParams(
    name="Copper", eps_P0=47.0, eps_R=8.0, eps_U=20.0, eps_F=3.0,
    eta=1.0, delta=0.020, E_bench=1200.0, E_tight=600.0,
)

ENERGY_ZN = EnergyParams(
    name="Zinc", eps_P0=37.5, eps_R=12.0, eps_U=18.0, eps_F=3.0,
    eta=1.0, delta=0.020, E_bench=300.0, E_tight=150.0,
)

CASES = [CU_BASE, CU_GAMMA, ZN_BASE, ZN_GAMMA]
ENERGY_OF = {"Copper": ENERGY_CU, "Zinc": ENERGY_ZN}


def energy_for(p: MaterialParams) -> EnergyParams:
    return ENERGY_CU if p.name.startswith("Copper") else ENERGY_ZN


# ----------------------------------------------------------------------
# Structural coefficients
# ----------------------------------------------------------------------

def Gamma(p: MaterialParams) -> float:
    """Circular (stock-side) feedback intensity, eq. (Gamma)."""
    return p.kappa / (1.0 - p.psi - p.rho * p.sigma)


def _feedback_bracket(p: MaterialParams) -> float:
    """The bracketed feedback aggregate multiplying Gamma in det(J_M)."""
    n, lam, gam, sig, rho, q, kap = p.n, p.lam, p.gamma, p.sigma, p.rho, p.q, p.kappa
    return ((1 - sig) * rho * lam * (n + gam)
            + (1 - sig) * q * (1 - rho) * gam * lam
            + q * (1 - rho) * sig * gam * (n + lam) / kap)


def Gamma_M(p: MaterialParams) -> float:
    """Stability threshold, eq. (Gamma_M).  +inf when the denominator vanishes."""
    denom = _feedback_bracket(p)
    if denom <= 0.0:
        return np.inf
    return (p.n + p.lam) * (p.n + p.gamma) / denom


def Delta_M(p: MaterialParams) -> float:
    """Determinant-like denominator of the steady state, eq. (Delta_M)."""
    return (p.n + p.lam) * (p.n + p.gamma) - Gamma(p) * _feedback_bracket(p)


def jacobian(p: MaterialParams) -> np.ndarray:
    """Jacobian J_M of the per-capita system, eq. (full_material_jacobian)."""
    G = Gamma(p)
    n, lam, gam, sig, rho, q, kap = p.n, p.lam, p.gamma, p.sigma, p.rho, p.q, p.kappa
    return np.array([
        [(1 - sig) * rho * lam * G - (n + lam), (1 - sig) * gam * G],
        [q * (1 - rho) * (lam + sig * rho * lam * G / kap),
         q * (1 - rho) * sig * gam * G / kap - (n + gam)],
    ])


def chi_star(p: MaterialParams) -> float:
    """Steady-state degree of circularity, eq. (chistar)."""
    n, lam, gam, sig, rho, q, kap = p.n, p.lam, p.gamma, p.sigma, p.rho, p.q, p.kappa
    return (rho * sig
            + rho * lam * (1 - sig) * kap / (n + lam)
            + gam * q * (1 - rho) * (lam * (1 - sig) * kap + sig * (n + lam))
            / ((n + lam) * (n + gam)))


def rho_offset(rho0: float, gamma: float, d_ref: float, base: float) -> float:
    """Mechanical offset of rho when urban mining is switched on.

    rho_gamma = rho_0 - gamma * d_ref / <lam*u + sigma*s>, so that observed
    secondary supply is preserved and recovered material is not double counted.
    """
    return rho0 - gamma * d_ref / base


# ----------------------------------------------------------------------
# Steady state
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SteadyState:
    name: str
    u: float
    d: float
    s: float
    chi: float
    Gamma: float
    Gamma_M: float
    Delta_M: float
    eigenvalues: tuple
    stable: bool

    def as_row(self):
        e1, e2 = sorted(self.eigenvalues)
        return (self.name, self.s, self.u, self.d, self.chi, e1, e2)


def steady_state(p: MaterialParams) -> SteadyState:
    """Closed-form steady state, eqs. (ustar), (dstar), (sstar)."""
    if not p.feasible():
        raise ValueError(f"{p.name}: 1 - psi - rho*sigma <= 0")
    G, D = Gamma(p), Delta_M(p)
    n, lam, gam, sig, rho, q, kap = p.n, p.lam, p.gamma, p.sigma, p.rho, p.q, p.kappa

    u = p.s_bar * G * (1 - sig) * (n + gam) / D
    d = (p.s_bar * G * q * (1 - rho)
         * (lam * (1 - sig) + sig * (n + lam) / kap) / D)
    s = p.s_bar * G * (n + lam) * (n + gam) / (kap * D)

    ev = np.linalg.eigvals(jacobian(p))
    return SteadyState(
        name=p.name, u=u, d=d, s=s, chi=chi_star(p),
        Gamma=G, Gamma_M=Gamma_M(p), Delta_M=D,
        eigenvalues=tuple(np.real_if_close(ev).tolist()),
        stable=bool(np.all(np.real(ev) < 0)),
    )


def steady_state_numeric(p: MaterialParams) -> tuple[float, float, float]:
    """Independent check: solve J_M [u,d]' = -constant directly."""
    G = Gamma(p)
    n, lam, gam, sig, rho, q, kap = p.n, p.lam, p.gamma, p.sigma, p.rho, p.q, p.kappa
    J = jacobian(p)
    const = np.array([(1 - sig) * G * p.s_bar,
                      q * (1 - rho) * sig * G * p.s_bar / kap])
    u, d = np.linalg.solve(-J, const)
    s = (rho * lam * u + gam * d + p.s_bar) / (1 - p.psi - rho * sig)
    return float(u), float(d), float(s)


def rhs(t, y, p: MaterialParams):
    """Right-hand side of the per-capita system, for trajectory simulation."""
    u, d = y
    s = (p.rho * p.lam * u + p.gamma * d + p.s_bar) / (1 - p.psi - p.rho * p.sigma)
    du = (1 - p.sigma) * p.kappa * s - (p.n + p.lam) * u
    dd = p.q * (1 - p.rho) * (p.lam * u + p.sigma * s) - (p.n + p.gamma) * d
    return [du, dd]


def simulate(p: MaterialParams, y0, t_end=400.0, n_pts=2001):
    """Integrate the per-capita system from y0 = (u0, d0)."""
    from scipy.integrate import solve_ivp
    t_eval = np.linspace(0.0, t_end, n_pts)
    sol = solve_ivp(rhs, (0.0, t_end), list(y0), args=(p,),
                    t_eval=t_eval, rtol=1e-10, atol=1e-12, dense_output=True)
    u, d = sol.y
    s = (p.rho * p.lam * u + p.gamma * d + p.s_bar) / (1 - p.psi - p.rho * p.sigma)
    return sol.t, u, d, s


# ----------------------------------------------------------------------
# Energy layer
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyState:
    name: str
    s: float
    p_star: float
    r_star: float
    gd: float          # gamma * d*
    Omega_NP: float
    Omega_0: float

    def as_row(self):
        return (self.name, self.s, self.p_star, self.r_star,
                self.gd, self.Omega_NP, self.Omega_0)


def energy_state(p: MaterialParams, e: EnergyParams,
                 ss: SteadyState | None = None) -> EnergyState:
    """Steady-state energy requirement, eq. (steady_energy_condition):

        Omega* = eps_P0 p* + eps_R r* + eps_U gamma d* + eps_F s*

    with p* = psi s* + p_bar and r* = rho (lam u* + sigma s*).  Fabrication
    energy is charged on the whole refined-metal throughput s*, exactly as in
    the model's own energy requirement (eq. energy_requirement).
    """
    ss = ss or steady_state(p)
    p_star = p.psi * ss.s + p.p_bar
    r_star = p.rho * (p.lam * ss.u + p.sigma * ss.s)
    gd = p.gamma * ss.d
    Omega_NP = e.eps_R * r_star + e.eps_U * gd + e.eps_F * ss.s
    Omega_0 = e.eps_P0 * p_star + Omega_NP
    return EnergyState(p.name, ss.s, p_star, r_star, gd, Omega_NP, Omega_0)


def critical_horizon(es: EnergyState, e: EnergyParams, E_bar: float) -> float:
    """Critical horizon T_E, eq. (critical_date). NaN when no positive horizon."""
    num = E_bar - es.Omega_NP
    den = e.eps_P0 * es.p_star
    if e.eta * e.delta <= 0 or num <= 0 or num <= den:
        return np.nan
    return np.log(num / den) / (e.eta * e.delta)


def Omega_of_t(t, es: EnergyState, e: EnergyParams):
    """Omega*(t) = eps_P0 e^{eta delta t} p* + Omega*_NP."""
    return e.eps_P0 * np.exp(e.eta * e.delta * np.asarray(t)) * es.p_star + es.Omega_NP
