# Dropping $\kappa$ from the fabrication-energy term

Four live occurrences. Each `FIND` string is unique in its file (checked with
`grep -c`), so each block is a safe single find/replace. The commented-out line
636 of `paper_final.tex` also carries the $\kappa$ and is left untouched.

After these edits the calibration equations agree with the model's own energy
requirement, eq.~\eqref{eq:energy_requirement}, which already charges
fabrication on the whole refined-metal flow. **No printed number changes**: the
tables were computed on $s^*$ all along.

---

## 1. `ch1_macro_materials.tex`, line 496 (Section 5.2, first sentence)

FIND
```
Using $\Omega^*=\varepsilon_{P0}p^*+\varepsilon_R r^*+\varepsilon_U\gamma d^*+\varepsilon_F\kappa s^*$, Table~\ref{tab2} reports
```

REPLACE
```
Using $\Omega^*=\varepsilon_{P0}p^*+\varepsilon_R r^*+\varepsilon_U\gamma d^*+\varepsilon_F s^*$, Table~\ref{tab2} reports
```

---

## 2. `ch1_macro_materials.tex`, line 1841 (eq. `oc_omega_growth`)

FIND
```
    +\varepsilon_F\kappa\mathcal I_S^*
```

REPLACE
```
    +\varepsilon_F\mathcal I_S^*
```

---

## 3. `app_ch1_calibration_details.tex`, line 461 (eq. `energy_calibration_app`)

FIND
```
    +\varepsilon_F\kappa S(t).
```

REPLACE
```
    +\varepsilon_F S(t).
```

---

## 4. `app_ch1_calibration_details.tex`, line 505 (eq. `nonprimary_energy_app`)

FIND
```
    +\varepsilon_F\kappa S^*.
```

REPLACE
```
    +\varepsilon_F S^*.
```

---

## Verification after applying

```
grep -n 'varepsilon_F\\kappa' ch1_macro_materials.tex app_ch1_calibration_details.tex app_ch1_trans_dyn.tex
```

should return nothing. `python run_all.py --no-figs` reproduces Table 3,
Table 4 and the observed-flow table unchanged.
