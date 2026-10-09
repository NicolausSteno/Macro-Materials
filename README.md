# Macro-Materials

Code for **Chapter 2, "Circularity, sustainability and growth under
biophysical constraints: A macro–materials framework"**, of the PhD thesis of
Nicola Gambaro (Imperial College London), and for its two appendices
(calibration data and estimation details; transitional dynamics).

The chapter builds a two-stock model of a metal in the economy, with primary
supply, scrap collection (ρ) and urban mining (γ). It derives the condition
for a stable positive steady state (Γ < Γ_M), the steady-state energy
requirement and the critical horizons at which it exceeds an energy budget,
and two extensions: a fixed-saving capital–material model and the
planner's intensity-sustainable optimum, including the circularity trap.
Everything is calibrated to copper and zinc.

The scripts recompute every number printed in the chapter and its appendices
from the source data and the model equations, check each one against the
value in the LaTeX, and build the figures.

## Quick start

Python 3.12 with the packages in `requirements.txt`:

```sh
pip install -r requirements.txt
python run_all.py             # tables, checks and figures
python run_all.py --no-figs   # tables and checks only
python mm_figures.py          # figures only  ->  ./figures/*.pdf, *.png
python oscillation_window.py  # closed-form oscillation criteria, verified
```

`run_all.py` prints a plain-text report and exits non-zero if any printed
value falls outside tolerance. `reproduction_report.txt` is the report of a
full run made with the zinc workbook present: every printed value
reproduces. The run takes under a minute on a laptop.

## Data availability

The zinc workbook `zinc_dataset_ORIGINAL.xlsx` (the annual reconstruction of
Rostek et al. 2022) contains International Lead and Zinc Study Group (ILZSG)
data that may not be redistributed, so it is **not included** in this
repository. Only section 1 of `run_all.py` uses it, to rebuild the zinc
parameters (ψ, P̄, ξ̄, λ, σ, κ, ρ₀, q, d) and the zinc row of the observed-flow
energy table from the annual series. If the file is absent, that part is
skipped with a notice.

Every other result runs as usual, because it uses the calibrated parameter
values stored in `mm_core.py`. This covers steady states, stability, energy
requirements, critical horizons, the ψ-sensitivity band, the growth and
optimal-control extensions, and all figures. The copper calibration in section
1 is built from the published stock and flow totals of Glöser, Soulier and
Tercero Espinoza (2013) and always runs. Readers with ILZSG access can place
the workbook in this folder to rerun the zinc reconstruction.

## What `run_all.py` reproduces

| Section of the report | Chapter content |
|---|---|
| 1. Calibration reconstructed from source data | Calibration appendix |
| 2. Calibrated steady states and local stability | Steady states and the stability condition |
| 3. Steady-state energy requirements and critical horizons | Energy requirements and critical horizons |
| 4. Sensitivity to the primary-supply slope | ψ-sensitivity band (calibration appendix) |
| 5. Capital–material extension: structural identities | Growth extension |
| 6. Intensity-sustainable optimum at copper parameters | Transitional-dynamics appendix |
| 7. Endogenous recovery capacity and the circularity trap | Transitional-dynamics appendix |

## Modules

| File | Contents |
|---|---|
| `mm_core.py` | Baseline two-stock model: `Gamma`, `Gamma_M`, `Delta_M`, `chi_star`, `jacobian`, closed-form `steady_state` (checked against a direct linear solve), the energy block `energy_state` and `critical_horizon`, and `simulate` for trajectories. Holds the calibrated parameter values. |
| `mm_calibration.py` | Zinc calibration rebuilt annually from the workbook; copper calibration from the Glöser et al. stock and flow totals; OLS for ψ and P̄; the q mapping; the observed-flow energy table |
| `mm_growth.py` | Fixed-saving capital–material extension: `k_star`, `jacobian3`, `routh_hurwitz`, the `decomposition` into τ_M, Π_M, 𝔦_k, F_M, G_M, the Schur identity, the Hawkins–Simon minors, `chi_g`, `balanced_growth_rate`, `detrended_jacobian`, comparative statics |
| `mm_optimal.py` | Detrended canonical state–costate system of the planner problem; analytic allocation first-order conditions; `solve_stationary`; `jacobian6`; `invariants` (the cubic K₁, K₂, K₃ and the ω roots); `dockner_ok`; the π and endogenous-ϱ extensions; `saddle_node_scan` and `find_fold` |
| `oscillation_window.py` | Stand-alone check of the oscillation criteria of the local-stability analysis: the exact reachability condition and an explicit sufficient window of coupling strengths, verified at the calibrated blocks and over a random sweep |
| `mm_figures.py` | All figures |
| `run_all.py` | Driver and check report |

Everything is analytic or solved to machine precision. Nothing is hard-coded
from the chapter except the printed values used as check targets.

## Figures

`mm_figures.py` writes eight figures to `./figures/`. Four of them appear in the
thesis, and these are published in the repository root under the file names
the thesis uses:

| File in this repository | Written by `mm_figures.py` as | What it shows |
|---|---|---|
| `ch1_stability_plane.pdf` | `fig_stability_plane.pdf` | The stability condition in the (ρ, γ) plane: the Γ < Γ_M region, the calibrated points and the γ = 0 comparison points for copper and zinc |
| `ch1_energy_horizon.pdf` | `fig_energy_horizon.pdf` | Ω\*(t) against the benchmark and tight energy budgets, with the critical horizons marked, calibrated against no urban mining |
| `ch1_investment_regimes.pdf` | `fig_investment_regimes.pdf` | The investment regimes as a bifurcation diagram: the three roots against 𝔦_k, and the regime map in (𝔦_k, τ_M) locating the explosive-oscillatory wedge |
| `ch1_skiba_fold.pdf` (appendix) | `fig_skiba_fold.pdf` | The circularity trap as a fold: the circular and middle configurations meeting in a saddle-node as m̄_R rises, with the linear corner persisting |

The other four (`fig_bifurcation_baseline`, `fig_dilution`, `fig_transition`,
`fig_dockner`) are diagnostic figures that the thesis does not use. They are
rebuilt by `mm_figures.py` but not published.

## Licence

Code: PolyForm Noncommercial License 1.0.0. Figures, results and
documentation: CC BY-NC 4.0, the licence of the thesis. Research, teaching and
other non-commercial use is free; commercial use needs a separate licence from
the author. Third-party data stay under their owners' terms. See
[`LICENSE`](LICENSE).

## Citation

If you use this code, please cite the thesis chapter. A Zenodo DOI for this
repository will be added here once the first release is archived.

## References

- Glöser, S., Soulier, M. and Tercero Espinoza, L. A. (2013). Dynamic analysis
  of global copper flows: global stocks, postconsumer material flows,
  recycling indicators, and uncertainty evaluation. *Environmental Science &
  Technology* 47(12), 6564–6572. <https://doi.org/10.1021/es400069b>
- Rostek, L., Tercero Espinoza, L. A., Goldmann, D. and Loibl, A. (2022). A
  dynamic material flow analysis of the global anthropogenic zinc cycle:
  providing a quantitative basis for circularity discussions. *Resources,
  Conservation and Recycling* 180, 106154.
  <https://doi.org/10.1016/j.resconrec.2022.106154>
