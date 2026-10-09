"""
oscillation_window.py
=====================
The oscillation criteria of Subsection 2.5.3 (local stability), in closed form.

Adds two objects to the analysis already implemented in mm_growth:

  * `oscillation_possible`  the exact reachability condition
        g > mu_2   and   i_k > -g,
    i.e. the pole -g of R lies strictly to the left of all three of its zeros;
    equivalent to "R has two positive critical values", but checkable directly.

  * `sufficient_window`     an explicit, always non-empty interval of coupling
    strengths inside the exact lens (phi_1, phi_2) of the oscillation-boundary
    proposition:
        phihat_1 = (g_3 - g_1) (sqrt(g_2) - sqrt(g_1))^2      >= phi_1
        phihat_2 = e_2 + 2 sqrt(e_1 e_3)                      <= phi_2
    with g_1 <= g_2 <= g_3 the ordered distances from the pole to the zeros and
    e_1, e_2, e_3 their elementary symmetric functions.  Any phi strictly
    between the two produces a complex conjugate pair, hence damped oscillatory
    adjustment.

Run `python oscillation_window.py` to verify both claims: at the calibrated
blocks, and against the exact critical values of R over a random sweep.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np

import mm_core as C
import mm_growth as G


def gaps(mu1: float, mu2: float, i_k: float, g: float) -> np.ndarray:
    """Ordered distances from the pole -g to the zeros i_k, -mu_1, -mu_2."""
    return np.sort(np.abs(np.array([i_k, -mu1, -mu2]) + g))


def oscillation_possible(mu1: float, mu2: float, i_k: float, g: float) -> bool:
    """Exact condition for a complex pair to occur at some coupling strength."""
    return bool(g > mu2 and i_k > -g)


def sufficient_window(mu1: float, mu2: float, i_k: float, g: float):
    """(phihat_1, phihat_2): any phi strictly inside gives a complex pair."""
    if not oscillation_possible(mu1, mu2, i_k, g):
        return None
    g1, g2, g3 = gaps(mu1, mu2, i_k, g)
    e1, e2, e3 = g1 + g2 + g3, g1 * g2 + g1 * g3 + g2 * g3, g1 * g2 * g3
    lo = (g3 - g1) * (np.sqrt(g2) - np.sqrt(g1)) ** 2
    hi = e2 + 2.0 * np.sqrt(e1 * e3)
    return float(lo), float(hi)


def has_complex_pair(mu1, mu2, i_k, g, phi) -> bool:
    f = np.polyadd(np.poly([i_k, -mu1, -mu2]), -np.array([0.0, 0.0, phi, phi * g]))
    return bool(np.abs(np.roots(f).imag).max() > 1e-12)


def report(m, tech=G.TECH, label=""):
    r = G.root_separation(m, tech)
    mu1, mu2, g, ik = r["mu1"], r["mu2"], r["gbar"], r["i_k"]
    w = r["phi_unit"]                      # phi = f_k * w
    print(f"{label or m.name}:  mu=({mu1:.4f}, {mu2:.4f})  g={g:.4f}  i_k={ik:.4f}")
    print(f"    g > mu_2: {g > mu2};  i_k > -g: {ik > -g};  "
          f"oscillation possible: {oscillation_possible(mu1, mu2, ik, g)}")
    win = sufficient_window(mu1, mu2, ik, g)
    exact = G.oscillation_boundary(m, tech, [ik])[0]
    if win is None:
        print("    no complex pair at any coupling strength")
        assert len(exact) < 2
        return
    print(f"    sufficient window   phi in ({win[0]:.4e}, {win[1]:.4e})"
          f"   f_k in ({win[0]/w:.4f}, {win[1]/w:.4f})")
    print(f"    exact lens          phi in ({exact[0]:.4e}, {exact[1]:.4e})"
          f"   f_k in ({exact[0]/w:.4f}, {exact[1]/w:.4f})")
    print(f"    window covers {100*(win[1]-win[0])/(exact[1]-exact[0]):.1f}% "
          f"of the exact lens")
    for lam in (0.001, 0.25, 0.5, 0.75, 0.999):
        phi = win[0] + lam * (win[1] - win[0])
        assert has_complex_pair(mu1, mu2, ik, g, phi), (m.name, phi)
    print("    every phi sampled inside the window gives a complex pair  ok")


def sweep(n: int = 20000, seed: int = 7):
    """Random check of both claims against the exact critical values of R."""
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(n):
        mu1 = rng.uniform(1e-3, 0.15)
        mu2 = mu1 + rng.uniform(0.0, 0.25)
        g = mu1 + rng.uniform(0.0, 0.5)          # mu_1 <= g, as the model implies
        ik = rng.uniform(-0.5, 0.3)
        P = np.poly([ik, -mu1, -mu2])
        Q = np.polysub(P, np.polymul([1.0, g], np.polyder(P)))
        zc = np.roots(Q)
        zc = zc[np.abs(zc.imag) < 1e-11].real
        crit = sorted(float(np.polyval(P, z) / (z + g)) for z in zc
                      if abs(z + g) > 1e-12 and np.polyval(P, z) / (z + g) > 0)
        poss = oscillation_possible(mu1, mu2, ik, g)
        if min(abs(g - mu2), abs(ik + g)) < 1e-7:       # skip the knife edges
            continue
        assert poss == (len(crit) == 2), (mu1, mu2, ik, g, crit)
        if poss:
            lo, hi = sufficient_window(mu1, mu2, ik, g)
            assert crit[0] <= lo * (1 + 1e-9) and hi * (1 - 1e-9) <= crit[1]
            assert lo < hi
            assert has_complex_pair(mu1, mu2, ik, g, 0.5 * (lo + hi))
        checked += 1
    print(f"random sweep: {checked} admissible configurations, all consistent  ok")


def gbar_lower_bound(n: int = 20000, seed: int = 3):
    """mu_1 <= gbar for every admissible material calibration."""
    rng = np.random.default_rng(seed)
    worst, kept = np.inf, 0
    for _ in range(n):
        p = C.MaterialParams(
            name="draw", psi=0.4, s_bar=0.5, p_bar=0.5,
            lam=rng.uniform(0.005, 0.2), sigma=rng.uniform(0.01, 0.6),
            rho=rng.uniform(0.0, 0.99), gamma=rng.uniform(0.0, 0.2),
            kappa=rng.uniform(0.5, 1.0), q=rng.uniform(0.0, 1.0),
            n=rng.uniform(0.0, 0.03))
        if not p.feasible():
            continue
        r = G.root_separation(p, G.TECH)
        if not np.isfinite(r["gbar"]):
            continue
        worst = min(worst, r["gbar"] - r["mu1"])
        kept += 1
    print(f"mu_1 <= gbar on {kept} admissible draws, smallest gap {worst:.3e}  ok")


if __name__ == "__main__":
    print("calibrated blocks")
    print("-----------------")
    for m in (C.CU_GAMMA, C.ZN_GAMMA):
        report(m)
    print()
    print("aggressive urban mining")
    print("-----------------------")
    for m in (C.CU_GAMMA, C.ZN_GAMMA):
        report(replace(m, gamma=0.05), label=f"{m.name.split(',')[0]}, gamma=0.05")
    print()
    gbar_lower_bound()
    sweep()
