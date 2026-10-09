# Chapter 2 — numerical reproduction and figures

Python scripts that recompute every number in `ch1_macro_materials.tex`,
`app_ch1_calibration_details.tex` and `app_ch1_trans_dyn.tex` from the source
data and the model equations, check each against the value printed in the
LaTeX, and build the figures.

```
python run_all.py             # tables, checks and figures
python run_all.py --no-figs   # tables and checks only
python mm_figures.py          # figures only  ->  ./figures/*.pdf, *.png
```

`run_all.py` exits non-zero if any printed value falls outside tolerance. As of
this run every printed value in both documents reproduces.

## Data availability

The zinc workbook `zinc_dataset_ORIGINAL.xlsx` (the annual reconstruction of
Rostek et al. 2022) contains ILZSG data that may not be redistributed, so it is
not included in this repository. It is used only by section 1 of `run_all.py`,
which rebuilds the zinc parameters (ψ, P̄, ξ̄, λ, σ, κ, ρ₀, q, d) and the zinc
row of the observed-flow energy table from the annual series. If the file is
absent, that part is skipped with a notice. Every other result runs as usual
because it uses the calibrated parameter values stored in `mm_core.py`: steady
states, stability, energy requirements, critical horizons, the ψ-sensitivity
band, the growth and optimal-control extensions, and all figures. The copper
calibration in section 1 is built from published Sankey totals and always runs.
`reproduction_report.txt` is a full run made with the workbook present.
Readers with ILZSG access can place the file in this folder to rerun the zinc
reconstruction.

## Modules

| file | contents |
|---|---|
| `mm_core.py` | baseline two-stock model: `Gamma`, `Gamma_M`, `Delta_M`, `chi_star`, `jacobian`, closed-form `steady_state` (checked against a direct linear solve), the energy block `energy_state` and `critical_horizon`, and `simulate` for trajectories |
| `mm_calibration.py` | zinc calibration rebuilt annually from `zinc_dataset_ORIGINAL.xlsx`; copper calibration from the Gloser et al. Sankey aggregates; OLS for `psi` and `Pbar`; the `q` mapping; the observed-flow energy table |
| `mm_growth.py` | fixed-saving capital–material extension: `k_star`, `jacobian3`, `routh_hurwitz`, the `decomposition` into `tau_M`, `Pi_M`, `i_k`, `F_M`, `G_M`, the Schur identity, the Hawkins–Simon minors, `chi_g`, `balanced_growth_rate`, `detrended_jacobian`, comparative statics |
| `mm_optimal.py` | detrended canonical state–costate system of the planner problem; analytic allocation FOCs; `solve_stationary`; `jacobian6`; `invariants` (the cubic `K_1, K_2, K_3` and the `omega` roots); `dockner_ok`; the `pi` and endogenous-`rho` extensions; `saddle_node_scan` and `find_fold` |
| `mm_figures.py` | all figures |
| `run_all.py` | driver and check report |

Everything is analytic or solved to machine precision; nothing is hard-coded
from the chapter except the printed values used as check targets.

## Figures (`./figures/`)

| file | what it shows |
|---|---|
| `fig_stability_plane.pdf` | Proposition 1 in the (ρ, γ) plane: the Γ < Γ_M region, the calibrated points, and the dotted calibration locus along which the ρ-offset rule moves the economy as urban mining is switched on |
| `fig_bifurcation_baseline.pdf` | the Γ = Γ_M boundary as a bifurcation: s\* blowing up as γ → γ_c at fixed ρ, and the dominant eigenvalue of J_M crossing zero at the same point — so the boundary of Proposition 1 is a zero-eigenvalue bifurcation, not a Hopf |
| `fig_dilution.pdf` | left, χ(g_s) falling towards ρσ as throughput growth rises; right, the unique crossing that defines the balanced growth rate on the knife edge |
| `fig_investment_regimes.pdf` | Table 5 as a bifurcation diagram: the three roots against 𝔦_k at the calibrated material block, and the two-parameter regime map in (𝔦_k, τ_M) locating the explosive-oscillatory wedge |
| `fig_energy_horizon.pdf` | Ω\*(t) against the benchmark and tight budgets, with all eight critical horizons marked |
| `fig_transition.pdf` | u, d, s and χ converging from the observed stocks to the calibrated steady state, with the half-life of the slow mode |
| `fig_dockner.pdf` | the three ω roots inside the region 𝒟, traced as the stock-dependent welfare weights are scaled from ×1 to ×20; and the six eigenvalues of J₆ in pairs summing to ϱ_e |
| `fig_skiba_fold.pdf` | the circularity trap as a fold: the circular and middle configurations approaching each other as m̄_R rises and annihilating in a saddle-node at m̄_R = 0.1421, with the linear corner persisting throughout |

