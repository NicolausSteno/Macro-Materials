"""
mm_figures.py
=============
All figures for Chapter 1 except the TikZ stock-flow diagram (fig_system.tex).

Every figure is built from the analytic expressions and the calibrated
parameters in mm_core / mm_growth / mm_optimal, so nothing here is drawn by
hand and each panel can be regenerated after a recalibration.

Figures produced
----------------
fig_stability_plane.pdf   Proposition 1 in the (rho, gamma) plane, with the
                          Gamma < Gamma_M region shaded and the calibrated
                          points marked.
fig_dilution.pdf          chi(g_s) and the growth dilution of circularity,
                          with the balanced-growth fixed point.
fig_energy_horizon.pdf    Omega*(t) against the benchmark and tight budgets,
                          with the critical horizons T_E.
fig_transition.pdf        Transition paths of u, d, s and chi from the
                          observed stocks to the calibrated steady state.
fig_dockner.pdf           The omega roots of the optimal system inside the
                          Dockner region D.
fig_bifurcation_baseline.pdf  Blow-up of the stationary state and the zero
                          eigenvalue at the Gamma = Gamma_M boundary.
fig_investment_regimes.pdf    The four local regimes of Table 5 as a
                          bifurcation diagram in net marginal investment.
fig_skiba_fold.pdf        The saddle-node in which the circular and middle
                          configurations annihilate: the circularity trap.

Run `python mm_figures.py` to write them all to ./figures/.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

import mm_core as C
import mm_growth as G
import mm_optimal as O

OUT = Path(__file__).with_name("figures")

# ----------------------------------------------------------------------
# House style: serif, thin rules, muted palette, no titles (LaTeX captions)
# ----------------------------------------------------------------------

CU, ZN = "#a1541c", "#1f4e79"
GREY, ACCENT = "#6b6b6b", "#5b2c8d"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.06,
})


def _save(fig, name: str):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


# ----------------------------------------------------------------------
# Figure: the stability plane
# ----------------------------------------------------------------------

def fig_stability_plane():
    """Gamma < Gamma_M in the (rho, gamma) plane, a picture of Proposition 1.

    Squares mark the calibration (gamma = 0.002 for copper, 0.003 for zinc,
    with the offset recycling coefficient), circles the no-urban-mining
    comparison at the observed coefficient rho_0.  The dotted line is the
    offset rule rho_gamma = rho_0 - gamma d_ref / <lam u + sigma s> that
    links the two as urban mining is switched on.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), sharey=False)
    cases = [(C.CU_BASE, C.CU_GAMMA, CU, "Copper", 1.902, 0.012),
             (C.ZN_BASE, C.ZN_GAMMA, ZN, "Zinc", 1.305, 0.030)]

    for ax, (base, sens, col, label, recyc_base, gmax) in zip(axes, cases):
        rho = np.linspace(0.01, 0.995, 320)
        gam = np.linspace(0.0, gmax, 320)
        R, Gm = np.meshgrid(rho, gam)
        margin = np.empty_like(R)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                q = replace(base, rho=R[i, j], gamma=Gm[i, j])
                margin[i, j] = C.Gamma_M(q) - C.Gamma(q) if q.feasible() else -1.0

        ax.contourf(R, Gm, (margin > 0).astype(float), levels=[0.5, 1.5],
                    colors=[col], alpha=0.13)
        ax.contour(R, Gm, margin, levels=[0.0], colors=[col], linewidths=1.3)

        # calibration locus
        gg = np.linspace(0.0, gmax, 200)
        rr = base.rho - gg * base.d_ref / recyc_base
        ok = rr > 0
        ax.plot(rr[ok], gg[ok], color=GREY, lw=1.0, ls=":")

        # circle: no-urban-mining comparison; square: calibration
        ax.plot(base.rho, 0.0, "o", color="white", ms=5, mec=col, mew=1.0, zorder=6)
        ax.plot(sens.rho, sens.gamma, "s", color=col, ms=5.5, mec="white", mew=0.8, zorder=6)
        ax.annotate(rf"$\gamma=0$, $\rho_0={base.rho:.3f}$", (base.rho, 0.0),
                    textcoords="offset points", xytext=(-7, 4), ha="right",
                    fontsize=7.5, color=col)
        # Keep the zinc calibration label clear of the y-axis.
        ax.annotate("calibrated:\n" + rf"$\gamma={sens.gamma:g}$, $\rho={sens.rho:.3f}$",
                    (sens.rho, sens.gamma), textcoords="offset points",
                    xytext=(-3, 4) if label == "Zinc" else (-6, 3),
                    ha="right", va="bottom",
                    fontsize=6.8 if label == "Zinc" else 7.5,
                    color=col, multialignment="right")

        ax.set_xlabel(r"scrap collection $\rho$")
        ax.set_ylabel(r"urban mining $\gamma$   (yr$^{-1}$)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, gmax)
        ax.text(0.03, 0.95, label, transform=ax.transAxes, fontsize=9,
                color=col, va="top")
        # In the zinc panel, move the stable-region label left of the calibration locus.
        gamma_left_x = 0.02 if label == "Zinc" else 0.14
        ax.text(gamma_left_x, 0.62 * gmax, r"$\Gamma<\Gamma_M$", fontsize=8, color=GREY)
        ax.text(0.80, 0.62 * gmax, r"$\Gamma>\Gamma_M$", fontsize=8, color=GREY)
    _save(fig, "fig_stability_plane")


# ----------------------------------------------------------------------
# Figure: the dilution of circularity by growth
# ----------------------------------------------------------------------

def fig_dilution():
    """chi(g_s) falls with throughput growth; the knife-edge fixed point."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
    g = np.linspace(0.0, 0.08, 400)

    ax = axes[0]
    for m, col, lab in ((C.CU_GAMMA, CU, "Copper"), (C.ZN_GAMMA, ZN, "Zinc")):
        chi = np.array([G.chi_g(m, gg) for gg in g])
        ax.plot(g, chi, color=col, label=lab)
        ax.plot(0, G.chi_g(m, 0), "o", color=col, ms=4.5, mec="white", mew=0.8)
        ax.axhline(m.rho * m.sigma, color=col, lw=0.7, ls=":")
        ax.annotate(rf"$\rho\sigma={m.rho*m.sigma:.3f}$", (0.078, m.rho * m.sigma),
                    fontsize=7.5, color=col, ha="right", va="bottom")
    ax.set_xlabel(r"throughput growth $g_s$   (yr$^{-1}$)")
    ax.set_ylabel(r"circularity $\chi(g_s)$")
    ax.set_xlim(0, 0.08)
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper right")

    # right panel: the balanced-growth fixed point on the knife edge
    ax = axes[1]
    t = G.knife_edge()
    gk = np.linspace(-0.005, 0.05, 500)
    for m, col, lab in ((C.CU_GAMMA, CU, "Copper"), (C.ZN_GAMMA, ZN, "Zinc")):
        rhs = np.array([G.g_map(x, m, t) + x for x in gk])
        ax.plot(gk, rhs, color=col, label=lab)
        gstar = G.balanced_growth_rate(m, t)
        ax.plot(gstar, gstar, "o", color=col, ms=4.5, mec="white", mew=0.8, zorder=5)
    ax.plot(gk, gk, color=GREY, lw=0.9, ls="--", label=r"45$^\circ$")
    ax.set_xlabel(r"capital growth $g_k$   (yr$^{-1}$)")
    ax.set_ylabel(r"$\iota\mathcal{M}(b_P,\chi(\vartheta g_k))-(n+\delta_K)$",
                  fontsize=8)
    ax.set_xlim(-0.005, 0.05)
    ax.set_ylim(-0.005, 0.05)
    ax.legend(loc="lower right")
    _save(fig, "fig_dilution")


# ----------------------------------------------------------------------
# Figure: the energy horizon
# ----------------------------------------------------------------------

def fig_energy_horizon():
    """Omega*(t) against the two energy budgets, with T_E marked."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), sharex=True)
    t = np.linspace(0, 110, 500)
    cases = [((C.CU_BASE, C.CU_GAMMA), C.ENERGY_CU, CU, "Copper", axes[0]),
             ((C.ZN_BASE, C.ZN_GAMMA), C.ENERGY_ZN, ZN, "Zinc", axes[1])]

    for (base, sens), e, col, label, ax in cases:
        ax.set_yscale("log")
        marks = []
        # solid: calibration (gamma > 0); dashed: no-urban-mining comparison
        for p_, ls, tag in ((sens, "-", "calibrated"), (base, "--", "no urban mining")):
            es = C.energy_state(p_, e)
            ax.plot(t, C.Omega_of_t(t, es, e), color=col, ls=ls,
                    label=rf"{tag}, $\gamma={p_.gamma:g}$")
            for E in (e.E_tight, e.E_bench):
                TE = C.critical_horizon(es, e, E)
                if np.isfinite(TE):
                    marks.append((TE, E, ls))
        lo = 0.5 * min(m[1] for m in marks) / 3
        for TE, E, ls in marks:
            ax.plot([TE, TE], [lo, E], color=col, lw=0.6, ls=":")
            ax.plot(TE, E, "o", color=col, ms=3.4, mec="white", mew=0.6, zorder=5)
        for E, name, xpos, ha in ((e.E_bench, "benchmark", 1.5, "left"),
                                  (e.E_tight, "tight", 108.5, "right")):
            ax.axhline(E, color=GREY, lw=0.8)
            ax.annotate(rf"$\bar E_m={E:g}$ ({name})", (xpos, E),
                        fontsize=7.5, color=GREY, ha=ha, va="bottom")
        ax.set_xlabel("years from 2026")
        ax.set_xlim(0, 110)
        ax.set_ylim(bottom=lo)
        ax.text(0.03, 0.95, label, transform=ax.transAxes, color=col, va="top")
        ax.legend(loc="lower right")

    axes[0].set_ylabel(r"$\Omega^*(t)$   (MJ cap$^{-1}$ yr$^{-1}$)")
    _save(fig, "fig_energy_horizon")


# ----------------------------------------------------------------------
# Figure: transition paths of the calibrated material system
# ----------------------------------------------------------------------

def fig_transition():
    """Convergence from the observed stocks to the calibrated steady state."""
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.6), sharex=True)
    for m, col, lab in ((C.CU_GAMMA, CU, "Copper"), (C.ZN_GAMMA, ZN, "Zinc")):
        ss = C.steady_state(m)
        tt, u, d, s = C.simulate(m, (m.u_ref, m.d_ref), t_end=400)
        chi = (m.rho * (m.lam * u + m.sigma * s) + m.gamma * d) / s
        for ax, series, star, ylab in (
            (axes[0, 0], u, ss.u, r"$u(t)$  (kg cap$^{-1}$)"),
            (axes[0, 1], d, ss.d, r"$d(t)$  (kg cap$^{-1}$)"),
            (axes[1, 0], s, ss.s, r"$s(t)$  (kg cap$^{-1}$ yr$^{-1}$)"),
            (axes[1, 1], chi, ss.chi, r"$\chi(t)$"),
        ):
            ax.plot(tt, series, color=col, label=lab)
            ax.axhline(star, color=col, lw=0.7, ls=":")
            ax.set_ylabel(ylab, fontsize=8.5)
            ax.margins(y=0.10)
        # half-life of the slow mode
        hl = np.log(2) / abs(max(np.real(ss.eigenvalues)))
        axes[1, 0].axvline(hl, color=col, lw=0.6, ls="--")
        axes[1, 0].annotate(f"half-life {hl:.0f} yr", (hl, 0.97),
                            xycoords=("data", "axes fraction"), xytext=(-3, 0),
                            textcoords="offset points",
                            fontsize=7, color=col, ha="right", va="top")
    for ax in axes[1]:
        ax.set_xlabel("years")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].set_xlim(0, 400)
    _save(fig, "fig_transition")


# ----------------------------------------------------------------------
# Figure: the omega roots inside the Dockner region
# ----------------------------------------------------------------------

def fig_dockner(scales=None):
    """The cubic criterion executed at copper parameters.

    Left: the three roots of eq. (oc_omega_cubic) inside the region D, traced
    as the three stock-dependent welfare weights are scaled up together.
    Right: the six eigenvalues of J_6, in pairs summing to the effective
    discount rate.
    """
    p = O.COPPER_OC
    scales = np.array([1, 2, 3, 5, 7.5, 10, 15, 20.]) if scales is None else scales
    pts, sols, g = [], {}, None
    for f in scales:
        sp = O.solve_stationary(
            replace(p, w_X=0.02 * f, w_U=0.01 * f, w_D=0.02 * f), guess=g)
        assert sp.residual < 1e-9, f"continuation failed at scale {f}"
        g = np.concatenate([sp.z, sp.w])
        pts.append(np.sort_complex(sp.omega))
        sols[f] = sp
    pts = np.array(pts)
    base, heavy = sols[scales[0]], sols[scales[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1))

    ax = axes[0]
    xlo, xhi, ylim = -0.013, 0.005, 0.0045
    im = np.linspace(-ylim, ylim, 400)
    ax.plot(im ** 2 / p.rho_e ** 2, im, color=GREY, lw=1.1)
    ax.fill_betweenx(im, xlo, im ** 2 / p.rho_e ** 2, color=GREY, alpha=0.10)
    for j in range(3):
        ax.plot(pts[:, j].real, pts[:, j].imag, color=CU, lw=0.7, alpha=0.6)
    ax.plot(pts[0].real, pts[0].imag, "o", color=CU, ms=5, mec="white",
            mew=0.8, ls="none", label="baseline weights")
    ax.plot(pts[-1].real, pts[-1].imag, "s", color=ACCENT, ms=5, mec="white",
            mew=0.8, ls="none", label=r"weights $\times$ 20")
    ax.axvline(0, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(r"$\mathrm{Re}\,\omega$")
    ax.set_ylabel(r"$\mathrm{Im}\,\omega$")
    ax.set_xticks([-0.012, -0.008, -0.004, 0.0, 0.004])
    ax.annotate(r"$\mathcal{D}$: one stable root per pair", (xlo * 0.93, 0.0033),
                fontsize=8, color=GREY)
    ax.legend(loc="lower left")

    ax = axes[1]
    for sp, col, mk, lab, yy in ((base, CU, "o", "baseline weights", 0.25),
                                 (heavy, ACCENT, "s", r"weights $\times$ 20", -0.25)):
        ev = np.sort(sp.eigenvalues.real)
        ax.plot(ev, np.full_like(ev, yy), mk, color=col, ms=5, mec="white",
                mew=0.8, ls="none", label=lab)
        for a, b in zip(ev[:3], ev[::-1][:3]):
            ax.plot([a, b], [yy, yy], color=col, lw=0.7, alpha=0.45)
    ax.axvline(0, color="k", lw=0.5)
    ax.axvline(p.rho_e / 2, color=GREY, lw=0.8, ls="--")
    ax.annotate(r"$\varrho_e/2$", (p.rho_e / 2, 0.80), fontsize=8, va="bottom",
                color=GREY, ha="center")
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"eigenvalues of $J_6$")
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right")
    _save(fig, "fig_dockner")


# ----------------------------------------------------------------------
# Figure: the circularity trap
# ----------------------------------------------------------------------

def fig_skiba_fold(scans=None):
    """Bifurcation diagram of the endogenous-recovery system.

    The linear corner exists at every parameterisation.  A circular
    configuration on the saturated branch and a middle configuration on the
    locally-increasing-returns branch coexist with it below a critical
    half-saturation scale, and annihilate in a saddle-node above it: the
    circularity trap.  Between the two interior points lies the Skiba threshold
    separating the basins of the linear and circular optima.
    """
    p = replace(O.COPPER_OC, rho_max=0.9, h=2.0, theta_R=0.6, m_bar=0.08)
    lin = O.solve_stationary_rho(p, "linear")
    g0 = np.concatenate([lin.z, lin.w])
    circ = O.solve_stationary_rho(p, "circular", guess=g0)
    mid = O.solve_stationary_rho(p, "middle", guess=g0)

    if scans is None:
        grid = np.concatenate([np.arange(0.080, 0.132, 0.004),
                               np.arange(0.132, 0.1425, 0.0005)])
        scans = {b: O.branch_scan(p, grid, b, guess=g0)
                 for b in ("circular", "middle")}
    arr = {b: np.array(v) for b, v in scans.items()}
    fold_lo, fold_hi = O.find_fold(p, 0.135, 0.150, guess=g0)

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7))

    # (a) the recovery technology and the three configurations
    ax = axes[0]
    mm = np.linspace(0, 0.26, 500)
    ax.plot(mm, [O.rho_hill(x, p)[0] for x in mm], color=CU)
    ax.axhline(p.rho_max, color=GREY, lw=0.8, ls="--")
    ax.axvline(O.inflection(p), color=GREY, lw=0.8, ls=":")
    for sp, mk, lab, col in ((circ, "o", "circular", CU),
                             (mid, "^", "middle", ACCENT)):
        S = sp.summary()
        ax.plot(S["mR"], S["rho"], mk, color=col, ms=5.5, mec="white", mew=0.8,
                zorder=5, label=lab)
    ax.plot(0, 0, "s", color=GREY, ms=5.5, mec="white", mew=0.8, zorder=5,
            label="linear")
    ax.annotate("inflection", (O.inflection(p), 0.86), fontsize=7,
                color=GREY, rotation=90, ha="right", va="top")
    ax.annotate(r"$\rho_{\max}$", (0.255, p.rho_max), fontsize=7.5,
                color=GREY, ha="right", va="bottom")
    ax.set_xlabel(r"recovery bundle $m_R$")
    ax.set_ylabel(r"$\rho(m_R)$")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")

    # (b) the fold in the recovery rate
    ax = axes[1]
    ax.annotate("linear corner", (0.081, 0.105), fontsize=7.5, color=GREY)
    ax.plot(arr["circular"][:, 0], arr["circular"][:, 1], color=CU, lw=1.4)
    ax.plot(arr["middle"][:, 0], arr["middle"][:, 1], color=ACCENT, lw=1.4,
            ls="--")
    ax.annotate("circular", (arr["circular"][6, 0], arr["circular"][6, 1]),
                textcoords="offset points", xytext=(3, 5), fontsize=7.5, color=CU)
    ax.annotate("middle (Skiba)", (arr["middle"][10, 0], arr["middle"][10, 1]),
                textcoords="offset points", xytext=(3, -11), fontsize=7.5,
                color=ACCENT)
    ax.plot([fold_lo], [0.5 * (arr["circular"][-1, 1] + arr["middle"][-1, 1])],
            "o", color="k", ms=4, zorder=6)
    ax.axvline(fold_lo, color=GREY, lw=0.7, ls=":")
    ax.annotate("saddle-node\n" rf"$\bar{{m}}_R={fold_lo:.4f}$",
                (fold_lo, 0.62), textcoords="offset points", xytext=(-6, 0),
                ha="right", fontsize=7.5)
    ax.set_xlabel(r"half-saturation scale $\bar{m}_R$")
    ax.set_ylabel(r"recovery rate $\rho^*$")
    ax.set_ylim(0.0, 0.7)

    # (c) the same fold in the recycling capital share
    ax = axes[2]
    for b, col, ls in (("circular", CU, "-"), ("middle", ACCENT, "--")):
        bR = []
        for mb in arr[b][:, 0]:
            sp = O.solve_stationary_rho(replace(p, m_bar=mb), b, guess=g0)
            bR.append(sp.summary()["bR"])
        ax.plot(arr[b][:, 0], bR, color=col, lw=1.4, ls=ls)
    ax.axvline(fold_lo, color=GREY, lw=0.7, ls=":")
    ax.set_xlabel(r"half-saturation scale $\bar{m}_R$")
    ax.set_ylabel(r"recycling capital share $b_R^*$")
    ax.set_ylim(0.0, 0.075)
    _save(fig, "fig_skiba_fold")


# ----------------------------------------------------------------------
# Figure: the baseline stability bifurcation
# ----------------------------------------------------------------------

def fig_bifurcation_baseline():
    """Blow-up of the stationary state as Gamma approaches Gamma_M.

    Left: the stationary throughput along the calibration locus in gamma.
    Right: the dominant eigenvalue crossing zero at the same point, so the
    boundary of Proposition 1 is a zero-eigenvalue bifurcation of the
    two-stock system rather than a Hopf.
    """
    from scipy.optimize import brentq
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
    for base, sens, col, lab, recyc_base, gmax in (
            (C.CU_BASE, C.CU_GAMMA, CU, "Copper", 1.902, 0.008),
            (C.ZN_BASE, C.ZN_GAMMA, ZN, "Zinc", 1.305, 0.020)):

        def at(gam):
            # rho held at its calibrated value: the offset rule of the
            # sensitivity lowers rho fast enough that the locus never reaches
            # the boundary, so the bifurcation is displayed at fixed rho.
            return replace(base, gamma=gam)

        crit = brentq(lambda g: C.Gamma(at(g)) - C.Gamma_M(at(g)),
                      1e-9, 0.5, xtol=1e-14)
        gg = np.linspace(0.0, min(gmax, 0.995 * crit), 400)
        s = np.array([C.steady_state(at(g)).s for g in gg])
        ev = np.array([max(np.real(C.steady_state(at(g)).eigenvalues)) for g in gg])

        for ax in axes:
            ax.axvline(crit, color=col, lw=0.8, ls="--")
            ax.axvline(sens.gamma, color=col, lw=0.7, ls=":")
            ax.annotate(rf"$\gamma_c={crit:.4f}$", (crit, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(-3, -2),
                        textcoords="offset points", ha="right", va="top",
                        rotation=90, fontsize=7.5, color=col)
        axes[0].plot(gg, s, color=col, label=lab)
        axes[0].annotate(rf"calibrated $\gamma={sens.gamma:g}$",
                         (sens.gamma, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(-3, -2), textcoords="offset points",
                         ha="right", va="top", rotation=90, fontsize=7,
                         color=col)
        axes[1].plot(gg, ev, color=col, label=lab)

    axes[0].set_xlabel(r"urban mining $\gamma$   (yr$^{-1}$)")
    axes[0].set_ylabel(r"stationary throughput $s^*$   (kg cap$^{-1}$ yr$^{-1}$)")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 0.02)
    axes[0].legend(loc="lower right")
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[1].set_xlabel(r"urban mining $\gamma$   (yr$^{-1}$)")
    axes[1].set_ylabel(r"dominant eigenvalue of $J_M$")
    axes[1].set_xlim(0, 0.02)
    axes[1].legend(loc="lower left")
    _save(fig, "fig_bifurcation_baseline")


# ----------------------------------------------------------------------
# Figure: the investment-behaviour bifurcation diagram
# ----------------------------------------------------------------------

def _regime(a1, a2, a3):
    """0 stable, 1 saddle, 2 explosive oscillation, 3 repeller."""
    ev = np.linalg.eigvals(np.array([[0.0, 0.0, -a3], [1.0, 0.0, -a2],
                                     [0.0, 1.0, -a1]]))
    npos = int((ev.real > 1e-15).sum())
    cplx = bool(np.any(abs(ev.imag) > 1e-14))
    if npos == 0:
        return 0
    if npos == 3:
        return 3
    if cplx and npos == 2:
        return 2
    return 1


_REGIME_LABEL = ["stable,\nmonotone", "stable, damped\noscillation",
                 "saddle,\nmonotone", "saddle,\noscillatory",
                 "explosive\noscillation", "repeller"]


# ----------------------------------------------------------------------
# Label placement: every label is kept strictly inside the region it names
# and clear of every drawn line; a leader line is used when it does not fit
# ----------------------------------------------------------------------

def _px(txt, fig):
    """Bounding box of a text artist in display coordinates."""
    return txt.get_window_extent(renderer=fig.canvas.get_renderer())


def _grow(bb, pad):
    return mpl.transforms.Bbox.from_extents(bb.x0 - pad, bb.y0 - pad,
                                            bb.x1 + pad, bb.y1 + pad)


def _clear(bb, blocked, curves_px, pad=3.0):
    """True if the box touches neither a previous label nor a drawn curve."""
    b = _grow(bb, pad)
    if any(b.overlaps(o) for o in blocked):
        return False
    if len(curves_px):
        hit = ((curves_px[:, 0] > b.x0) & (curves_px[:, 0] < b.x1)
               & (curves_px[:, 1] > b.y0) & (curves_px[:, 1] < b.y1))
        if hit.any():
            return False
    return True


def _deepest(mask, iks, fks, ax):
    """Point of `mask` farthest from its boundary, in data coordinates."""
    if not mask.any():
        return None
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    box = ax.get_window_extent()
    sx = box.width / (x1 - x0) * (iks[1] - iks[0])     # px per grid column
    sy = box.height / (y1 - y0) * (fks[1] - fks[0])    # px per grid row
    pad = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), bool)
    pad[1:-1, 1:-1] = mask
    dist = distance_transform_edt(pad, sampling=(sy, sx))[1:-1, 1:-1]
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    return float(iks[j]), float(fks[i])


def _inside(bb, mask, iks, fks, ax, pad=3.0):
    """True if the whole padded box lies inside the region."""
    inv = ax.transData.inverted()
    b = _grow(bb, pad)
    for x in (b.x0, b.x1):
        for y in (b.y0, b.y1):
            dx, dy = inv.transform((x, y))
            j = int(np.clip(np.searchsorted(iks, dx), 0, len(iks) - 1))
            i = int(np.clip(np.searchsorted(fks, dy), 0, len(fks) - 1))
            if not mask[i, j]:
                return False
    return True


_CAND = [(fx, fy) for fy in (0.94, 0.78, 0.60, 0.42, 0.24, 0.08)
         for fx in (0.07, 0.20, 0.33, 0.46, 0.59, 0.72, 0.85)]


def _leader(ax, fig, target, text, curves_px, blocked, col=GREY, fs=7.5):
    """Label in free space with a thin line pointing at the region."""
    best, tgt = None, np.array(ax.transData.transform(target))
    for fx, fy in _CAND:
        t = ax.text(fx, fy, text, transform=ax.transAxes, fontsize=fs,
                    color=col, ha="center", va="center", linespacing=1.15)
        fig.canvas.draw()
        bb = _px(t, fig)
        t.remove()
        if not _clear(bb, blocked, curves_px, pad=4.0):
            continue
        c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
        d = float(np.hypot(*(c - tgt)))
        if best is None or d < best[0]:
            best = (d, (fx, fy))
    xy = best[1] if best else (0.5, 0.5)
    ann = ax.annotate(text, xy=target, xycoords="data", xytext=xy,
                      textcoords="axes fraction", fontsize=fs, color=col,
                      ha="center", va="center", linespacing=1.15, zorder=7,
                      arrowprops=dict(arrowstyle="-", lw=0.55, color=col,
                                      shrinkA=1.5, shrinkB=1.5))
    fig.canvas.draw()
    blocked.append(_grow(_px(ann, fig), 2.0))


def _label_regions(ax, fig, Z, iks, fks, curves_px, blocked, col=GREY, fs=7.5):
    """One label per regime: inside the region if it fits, else a leader line."""
    fig.canvas.draw()
    for code in range(6):
        mask = Z == code
        if mask.sum() < 0.004 * mask.size:
            continue
        pt = _deepest(mask, iks, fks, ax)
        if pt is None:
            continue
        txt = ax.text(pt[0], pt[1], _REGIME_LABEL[code], fontsize=fs, color=col,
                      ha="center", va="center", linespacing=1.15, zorder=7)
        fig.canvas.draw()
        bb = _px(txt, fig)
        if (_clear(bb, blocked, curves_px, pad=2.5)
                and _inside(bb, mask, iks, fks, ax, pad=2.5)):
            blocked.append(_grow(bb, 2.0))
            continue
        txt.remove()
        _leader(ax, fig, pt, _REGIME_LABEL[code], curves_px, blocked,
                col=col, fs=fs)


def _curve_points(ax, xs, ys):
    """Sampled display coordinates of a drawn curve, for collision tests."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        return np.empty((0, 2))
    return ax.transData.transform(np.column_stack([xs[ok], ys[ok]]))


def _try_text(ax, fig, options, text, curves_px, blocked, **kw):
    """Place `text` at the first candidate whose box is clear of every line."""
    for opt in options:
        o = dict(kw)
        o.update(opt)
        xy = (o.pop("x"), o.pop("y"))
        t = ax.text(xy[0], xy[1], text, **o)
        fig.canvas.draw()
        bb = _px(t, fig)
        if _clear(bb, blocked, curves_px, pad=2.5):
            blocked.append(_grow(bb, 2.0))
            return t
        t.remove()
    o = dict(kw)
    o.update(options[-1])
    xy = (o.pop("x"), o.pop("y"))
    t = ax.text(xy[0], xy[1], text, **o)
    fig.canvas.draw()
    blocked.append(_grow(_px(t, fig), 2.0))
    return t


def _regime_plane(ax, m, iks, fks, title, col, panel=""):
    """Regime map in the capital-side plane (i_k, f_k) at a fixed material block.

    Both F_M and G_M are proportional to f_k while tau_M and Pi_M belong to the
    material block, so this plane spans every configuration a generalised
    investment schedule with non-negative marginal propensity can produce.
    """
    Z = np.empty((len(fks), len(iks)))
    for i, fk in enumerate(fks):
        c1, c2, c3 = G.coeffs_from_capital(m, G.TECH, iks, fk)
        for j in range(len(iks)):
            Z[i, j] = G.regime_code(c1[j], c2[j], c3[j])
    cmap = mpl.colors.ListedColormap([mpl.colors.to_rgba(col, 0.10),
                                      mpl.colors.to_rgba(col, 0.36),
                                      mpl.colors.to_rgba(GREY, 0.13),
                                      mpl.colors.to_rgba(GREY, 0.36),
                                      mpl.colors.to_rgba(ACCENT, 0.45),
                                      mpl.colors.to_rgba("k", 0.55)])
    ax.pcolormesh(iks, fks, Z, cmap=cmap, vmin=-0.5, vmax=5.5, shading="auto")
    ax.set_xlim(iks[0], iks[-1])
    ax.set_ylim(fks[0], fks[-1])

    fig = ax.figure
    curves = []
    # a_3 = 0, the stable/saddle boundary, is exactly linear in this plane
    dd = G.decomposition(m, G.TECH)
    PhiF, PhiG = dd["F_M"] / dd["f_k"], dd["G_M"] / dd["f_k"]
    xa3 = -fks * PhiG / dd["Pi_M"]
    ax.plot(xa3, fks, color="k", lw=1.1)
    curves.append((xa3, fks))
    # the oscillation boundary is the set of critical values of R, in closed form
    r = G.root_separation(m, G.TECH)
    grid = np.linspace(iks[0], iks[-1], 900)
    # the two branches meet with a cusp at i_k = -gbar, which a uniform grid
    # cannot resolve, so cluster extra abscissae just to the right of it
    if iks[0] < -r["gbar"] < iks[-1]:
        tip = -r["gbar"] + np.logspace(-9.0, -2.0, 240) * (iks[-1] - iks[0])
        grid = np.unique(np.concatenate([grid, tip[tip <= iks[-1]]]))
    B = G.oscillation_boundary(m, G.TECH, grid)
    two = np.array([len(v) == 2 for v in B])
    if two.any():
        g2 = grid[two]
        lo = np.array([v[0] / PhiF for v, t in zip(B, two) if t])
        hi = np.array([v[1] / PhiF for v, t in zip(B, two) if t])
        # one polyline down the lower branch and back up the upper one, so the
        # two are joined at the cusp instead of ending in a gap there
        xb = np.concatenate([g2[::-1], g2])
        yb = np.concatenate([lo[::-1], hi])
        vis = np.where(yb <= fks[-1], yb, np.nan)
        ax.plot(xb, vis, color="k", lw=1.1)
        curves.append((xb, vis))

    fig.canvas.draw()
    curves_px = np.vstack([_curve_points(ax, x, y) for x, y in curves])

    # panel key and header block, nudged down until clear of every line
    blocked = []
    if panel:
        key = ax.text(-0.01, 1.06, panel, transform=ax.transAxes, fontsize=9.5,
                      fontweight="bold", ha="left", va="bottom")
        fig.canvas.draw()
        blocked.append(_grow(_px(key, fig), 2.0))

    inside = r"\in" if r["separated"] else r"\notin"
    head = [title, rf"$\mathfrak{{g}}={r['gbar']:.4f}\;{inside}\;[\mu_1,\mu_2]$"]
    colours = [col, GREY]
    ytop = 0.975
    for line, c in zip(head, colours):
        for _ in range(24):
            t = ax.text(0.025, ytop, line, transform=ax.transAxes, color=c,
                        va="top", ha="left", fontsize=7.5, zorder=7)
            fig.canvas.draw()
            bb = _px(t, fig)
            if _clear(bb, blocked, curves_px, pad=3.0):
                blocked.append(_grow(bb, 2.0))
                ytop = ax.transAxes.inverted().transform((bb.x0, bb.y0))[1] - 0.022
                break
            t.remove()
            ytop -= 0.045
        else:
            ax.text(0.025, ytop, line, transform=ax.transAxes, color=c,
                    va="top", ha="left", fontsize=7.5, zorder=7)
            ytop -= 0.055

    _label_regions(ax, fig, Z, iks, fks, curves_px, blocked)

    ax.set_xlabel(r"net marginal investment $\mathfrak{i}_k$")
    return Z, r, curves_px, blocked


def fig_investment_regimes():
    """Which rows of Table 5 the model can reach.

    Panel a) roots against net marginal investment at the calibrated copper
    block.  By the Schur identity the boundary a_3 = 0 sits at i_k = -G_M/Pi_M,
    which is constant returns alpha + vartheta nu = 1.

    Panels b) and c) the capital-side plane (i_k, f_k).  Since tau_M and Pi_M
    are properties of the material block and both F_M and G_M are proportional
    to f_k, this plane spans every configuration a generalised investment
    schedule can produce at a given material calibration.  At the calibrated
    copper block the separating rate gbar = G_M/F_M lies strictly between the
    two material decay rates, so the roots are real everywhere and only the
    monotone rows occur.  Raising urban mining to gamma = 0.05 pushes gbar
    above mu_2 and opens the two oscillatory regions; panel c) is set on its
    own row so that both the header block and the four regime labels sit
    clear of the boundaries they separate.
    """
    d = G.decomposition(C.CU_GAMMA, G.TECH)
    Pi, GM, tau0, ik0 = d["Pi_M"], d["G_M"], d["tau_M"], d["i_k"]
    saddle = -GM / Pi

    fig = plt.figure(figsize=(7.4, 5.3))
    fig.set_layout_engine("none")
    gs_top = fig.add_gridspec(1, 2, left=0.085, right=0.985,
                              bottom=0.603, top=0.945, wspace=0.26)
    # panel c) is narrower than the top row but centred on it
    half = 0.310
    gs_bot = fig.add_gridspec(1, 1, left=0.535 - half, right=0.535 + half,
                              bottom=0.085, top=0.454)
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_bot[0, 0])

    # ---- a) roots against net marginal investment ----------------------
    ax = ax_a
    ik = np.linspace(-0.05, 0.02, 1200)
    a1, a3 = tau0 - ik, -ik * Pi - GM
    a2 = Pi - ik * tau0 - d["F_M"]
    re = np.array([np.sort(np.roots([1.0, x, y, z]).real)
                   for x, y, z in zip(a1, a2, a3)])
    ylo, yhi = -0.052, 0.027
    ax.axvspan(ik[0], saddle, color=CU, alpha=0.12)
    ax.axvspan(saddle, ik[-1], color=GREY, alpha=0.20)
    for j in range(3):
        ax.plot(ik, re[:, j], color=CU, lw=1.2)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(saddle, color="k", lw=0.9, ls="--")
    ax.axvline(ik0, color=ACCENT, lw=1.1)
    ax.set_xlabel(r"net marginal investment $\mathfrak{i}_k$")
    ax.set_ylabel(r"$\mathrm{Re}\,\zeta_i$")
    ax.set_xlim(ik[0], ik[-1])
    ax.set_ylim(ylo, yhi)
    ax.text(-0.01, 1.06, "a)", transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", ha="left", va="bottom")

    # every drawn line of the panel, for the collision tests below
    fig.canvas.draw()
    curves_a = [_curve_points(ax, ik, re[:, j]) for j in range(3)]
    curves_a += [_curve_points(ax, ik, np.zeros_like(ik)),
                 _curve_points(ax, np.full(200, saddle), np.linspace(ylo, yhi, 200)),
                 _curve_points(ax, np.full(200, ik0), np.linspace(ylo, yhi, 200))]
    curves_a = np.vstack(curves_a)
    blocked_a = []

    # the calibrated marker goes to the very top, in the empty strip above the
    # largest root; the a_3 = 0 key goes to the empty strip at the bottom
    _try_text(ax, fig, [dict(x=ik0 + 0.0016, y=yhi - 0.0015, ha="left", va="top"),
                        dict(x=ik0 - 0.0016, y=yhi - 0.0015, ha="right", va="top"),
                        dict(x=ik0 + 0.0016, y=yhi - 0.0015, ha="left", va="top",
                             fontsize=7.0)],
              "calibrated\ncopper", curves_a, blocked_a,
              fontsize=7.5, color=ACCENT, linespacing=1.15, zorder=7)
    key = ax.annotate("$a_3=0$:\n$\\alpha+\\vartheta\\nu=1$", xy=(saddle, ylo + 0.0075),
                      xytext=(0.0125, ylo + 0.0075), textcoords="data",
                      fontsize=7.5, color="k", ha="center", va="center",
                      linespacing=1.15, zorder=7,
                      arrowprops=dict(arrowstyle="-", lw=0.55, color="k",
                                      shrinkA=2, shrinkB=1))
    fig.canvas.draw()
    blocked_a.append(_grow(_px(key, fig), 2.0))
    _try_text(ax, fig, [dict(x=0.42, y=0.97), dict(x=0.34, y=0.97),
                        dict(x=0.52, y=0.97)],
              "stable", curves_a, blocked_a, transform=ax.transAxes,
              fontsize=8.5, color=GREY, va="top", ha="center")
    _try_text(ax, fig, [dict(x=0.955, y=0.97), dict(x=0.955, y=0.86)],
              "saddle", curves_a, blocked_a, transform=ax.transAxes,
              fontsize=8.5, color=GREY, va="top", ha="right")

    # ---- b) capital-side plane at the calibrated block ------------------
    iks = np.linspace(-0.12, 0.02, 320)
    fks = np.linspace(0.0, 0.06, 320)
    Z, r, curves_b, blocked_b = _regime_plane(ax_b, C.CU_GAMMA, iks, fks,
                                              r"copper, $\gamma=0.002$", CU,
                                              panel="b)")
    ax_b.plot(ik0, d["f_k"], "o", color=ACCENT, ms=5.5, mec="white", mew=0.8,
              zorder=8)
    ax_b.annotate("calibrated", (ik0, d["f_k"]), textcoords="offset points",
                  xytext=(-7, 4), fontsize=7.5, color=ACCENT, ha="right",
                  va="bottom", zorder=8)
    ax_b.set_ylabel(r"capital--material feedback $\mathfrak{f}_k$")

    # ---- c) the same plane with aggressive urban mining -----------------
    m2 = replace(C.CU_GAMMA, gamma=0.05)
    iks2 = np.linspace(-0.30, 0.02, 320)
    fks2 = np.linspace(0.0, 0.10, 320)
    Z2, r2, curves_c, blocked_c = _regime_plane(ax_c, m2, iks2, fks2,
                                                r"copper, $\gamma=0.05$", CU,
                                                panel="c)")
    ax_c.set_ylabel(r"capital--material feedback $\mathfrak{f}_k$")
    _save(fig, "fig_investment_regimes")


# ----------------------------------------------------------------------

ALL = [fig_stability_plane, fig_bifurcation_baseline, fig_dilution,
       fig_investment_regimes, fig_energy_horizon, fig_transition,
       fig_dockner, fig_skiba_fold]


if __name__ == "__main__":
    for f in ALL:
        print(f"building {f.__name__} ...")
        f()
