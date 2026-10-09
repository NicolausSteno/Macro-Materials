"""
mm_calibration.py
=================
Reconstruction of the calibrated parameters of Chapter 2 from source data.

Zinc  : rebuilt annually from `zinc_dataset.xlsx` (or the legacy
        `zinc_dataset_ORIGINAL.xlsx`), the reconstruction of
        Rostek et al. (2022), over 1980-2019.
Copper: rebuilt from the aggregate 2000-2010 Sankey totals of Gloser et al.
        (2013) as transcribed in the calibration appendix.

Reproduces:
  * the inventory term xi and its window mean          (eq. zinc_xi, zinc_xi_average_app)
  * the OLS calibration of psi and Pbar                (eq. psi_ols_app, barp_ols_app)
  * lambda, sigma, kappa, rho_0                        (eq. zinc_lambda_sigma_omega_app,
                                                        zinc_rho_def_app, zinc_rho0_app)
  * the recoverable-discard share q                    (eq. q_calibration_app)
  * the discard stock proxy d                          (eq. zinc_d_stock, copper_landfill_stock)
  * the rho offset when urban mining is switched on    (eq. zinc_rhogamma_app,
                                                        copper_rho_values_app)
  * the observed-flow energy table                     (Table observed_energy_app)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

_HERE = Path(__file__).resolve().parent
XLSX = (_HERE / "zinc_dataset.xlsx"
        if (_HERE / "zinc_dataset.xlsx").exists()
        else _HERE / "zinc_dataset_ORIGINAL.xlsx")

# The zinc workbook contains ILZSG data that may not be redistributed, so it
# is not part of the public repository.  Without it, only the zinc
# reconstruction (section 1 of run_all.py) is skipped: every model result
# uses the calibrated parameter values stored in mm_core.
HAVE_ZINC_DATA = XLSX.exists()

ZN_WINDOW = (1980, 2019)
CU_WINDOW = (2000, 2010)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_zinc(path: Path | str = XLSX) -> dict[str, pd.DataFrame]:
    """Load the four data sheets of the zinc workbook, indexed by year."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    out = {}
    for sheet in ("Flows", "Stocks", "Exogenous", "econ_cal"):
        rows = list(wb[sheet].iter_rows(values_only=True))
        d = pd.DataFrame(rows[1:], columns=rows[0]).dropna(subset=["Year"])
        d["Year"] = d["Year"].astype(int)
        out[sheet] = d.set_index("Year").astype(float)
    return out


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Simple OLS slope and intercept, eqs. (psi_ols_app)-(barp_ols_app)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    slope = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    return float(slope), float(ym - slope * xm)


# ----------------------------------------------------------------------
# Zinc
# ----------------------------------------------------------------------

def zinc_calibration(data: dict[str, pd.DataFrame] | None = None) -> dict:
    """Full zinc calibration over 1980-2019.  All flows in Mt/yr, stocks in Mt."""
    data = data or load_zinc()
    F, St, Ex = data["Flows"], data["Stocks"], data["Exogenous"]
    yrs = range(ZN_WINDOW[0], ZN_WINDOW[1] + 1)

    kt = 1e-3  # workbook is in kt
    N = Ex.loc[yrs, "Population"].to_numpy()

    # --- usable supply, eq. (zinc_S_app): S = FU + NS^FU + L^FU -------------
    S = (F.loc[yrs, "First use manufacturing"]
         + F.loc[yrs, "New scrap (first use)"]
         + F.loc[yrs, "First use manufacturing losses"]).to_numpy() * kt
    P = F.loc[yrs, "Primary Refining"].to_numpy() * kt

    # --- inventory term, eq. (zinc_xi) --------------------------------------
    xi = -(St.loc[yrs, "Change in Scrap Stock"]
           + St.loc[yrs, "Change in Refined Stock"]
           + St.loc[yrs, "Change in Concentrate Stock"]).to_numpy() * kt

    # --- OLS on aggregate series, eqs. (psi_ols_app)-(barp_ols_app) ---------
    psi, P_bar = ols(S, P)
    S_bar = P_bar + xi.mean()
    N_mean = N.mean()

    # --- stock-flow parameters, eq. (zinc_lambda_sigma_omega_app) ----------
    U = St.loc[yrs, "In-Use Stock"].to_numpy() * kt
    EOL = F.loc[yrs, "End of life"].to_numpy() * kt
    OS = F.loc[yrs, "Old scrap (i.e. separated EoL scrap)"].to_numpy() * kt
    W = F.loc[yrs, "Waelz process recycling"].to_numpy() * kt
    DR = F.loc[yrs, "Direct reuse recycling"].to_numpy() * kt
    XD = F.loc[yrs, "Dissipative use"].to_numpy() * kt

    lam_t = data["econ_cal"].loc[yrs, "lambda"].to_numpy()
    sig_t = data["econ_cal"].loc[yrs, "sigma"].to_numpy()
    kap_t = 1.0 - XD / ((1.0 - sig_t) * S)
    rho_t = (W + DR) / (EOL + sig_t * S)

    # --- recoverable share and discard stock, eqs. (q_zn_app)-(zinc_d_stock)
    # EOL - OS is the broad *unseparated EoL* quantity used only to construct
    # an upper bound on recoverability.  D itself must be consistent with the
    # model law dD/dt = q(1-rho)(EOL + new scrap) - gamma D.
    D_unseparated_eol = float((EOL - OS).sum())
    resid = float(((1.0 - rho_t) * (EOL + sig_t * S)).sum())
    q_upper = D_unseparated_eol / resid
    q = 0.300
    D_recoverable = q * resid
    d_2019 = D_recoverable / N[-1] * 1e9  # Mt -> kg/cap

    # --- recyclable base for the rho offset, eq. (zinc_rhogamma_app) -------
    base_t = (EOL + sig_t * S) / N * 1e9      # kg/cap/yr
    u_t = U / N * 1e9                          # kg/cap
    s_t = S / N * 1e9                          # kg/cap/yr
    p_t = P / N * 1e9                          # kg/cap/yr
    r_t = (W + DR) / N * 1e9                   # kg/cap/yr

    return dict(
        window=ZN_WINDOW, years=np.array(list(yrs)),
        S=S, P=P, xi=xi, N=N, U=U, EOL=EOL, OS=OS, W=W, DR=DR,
        S_mean=float(S.mean()), P_mean=float(P.mean()),
        psi=psi, P_bar=P_bar, xi_bar=float(xi.mean()), S_bar=S_bar,
        N_mean=float(N_mean), N_end=float(N[-1]),
        p_bar=P_bar / N_mean * 1e9,
        s_bar=S_bar / N_mean * 1e9,
        xi_bar_pc=xi.mean() / N_mean * 1e9,
        lam=float(lam_t.mean()), sigma=float(sig_t.mean()),
        kappa=float(kap_t.mean()), rho_0=float(rho_t.mean()),
        rho_t=rho_t, lam_t=lam_t, sig_t=sig_t, kap_t=kap_t,
        D_cum=D_unseparated_eol, D_recoverable=D_recoverable, d_2019=d_2019,
        resid=resid, q_upper=q_upper, q=q,
        base_mean=float(base_t.mean()),
        u_mean=float(u_t.mean()), s_mean=float(s_t.mean()),
        p_mean=float(p_t.mean()), r_mean=float(r_t.mean()),
    )


# ----------------------------------------------------------------------
# Copper (aggregate Sankey totals, Gloser et al. 2013)
# ----------------------------------------------------------------------

CU_SANKEY = dict(
    s_semis=241.7,       # Mt, semis production 2000-2010
    L_semis=1.2,         # Mt
    L_fab=1.2,           # Mt
    new_scrap=39.0,      # Mt
    EOL=101.0,           # Mt, end-of-life scrap
    into_use=201.5,      # Mt, flow into in-use stock
    U_2010=350.0,        # Mt, in-use stock at end of window
    collection_losses=35.0,   # Mt
    D_landfill=133.0,    # Mt, landfill stock
    P_obs=162.0,         # Mt, primary refined copper
    rho_0=0.608,
    xi_bar=-0.15,        # Mt/yr, inventory adjustment
    years=11,
)


def copper_calibration(c: dict | None = None) -> dict:
    """Copper calibration from the aggregate 2000-2010 Sankey totals."""
    c = c or CU_SANKEY
    S_tot = c["s_semis"] + c["L_semis"] + c["L_fab"]            # 244.1 Mt
    sigma = (c["new_scrap"] + c["L_semis"] + c["L_fab"]) / S_tot
    lam = (c["EOL"] / c["years"]) / c["U_2010"]
    kappa = c["into_use"] / ((1 - sigma) * S_tot)

    NS = c["new_scrap"] + c["L_semis"] + c["L_fab"]
    resid = (1 - c["rho_0"]) * (c["EOL"] + NS)
    q = c["collection_losses"] / resid

    # populations implied by the reported per-capita values
    d_ref = 19.3                     # kg/cap, reported landfill stock
    u_ref = 50.7                     # kg/cap, reported in-use stock
    N_2010 = c["U_2010"] / u_ref * 1e9          # persons
    s_obs = 3.375                    # kg/cap/yr, window-average throughput
    N_win = S_tot / c["years"] / s_obs * 1e9

    psi, P_bar = 0.474, 4.090        # OLS on the adjusted annual series
    p_bar = P_bar / N_win * 1e9
    s_bar = p_bar + c["xi_bar"] / N_win * 1e9

    base = lam * u_ref + sigma * s_obs

    return dict(
        window=CU_WINDOW, S_tot=S_tot, sigma=sigma, lam=lam, kappa=kappa,
        q=q, resid=resid, rho_0=c["rho_0"],
        d_ref=d_ref, u_ref=u_ref, s_obs=s_obs,
        N_2010=N_2010, N_win=N_win,
        psi=psi, P_bar=P_bar, p_bar=p_bar, s_bar=s_bar,
        xi_bar_pc=c["xi_bar"] / N_win * 1e9,
        base=base,
        D_implied_from_d=d_ref * N_2010 / 1e9,
        p_obs=c["P_obs"] / c["years"] / N_win * 1e9,
        r_obs=(S_tot - c["P_obs"]) / c["years"] / N_win * 1e9,
        r_obs_model=c["rho_0"] * (lam * u_ref + sigma * s_obs),
    )


# ----------------------------------------------------------------------
# Observed-flow energy table
# ----------------------------------------------------------------------

def observed_energy(zn: dict, cu: dict,
                    eps_cu=(47.0, 8.0, 3.0), eps_zn=(37.5, 12.0, 3.0)) -> pd.DataFrame:
    """Energy implied by observed calibration-window flows (Table observed_energy_app)."""
    ePc, eRc, eFc = eps_cu
    ePz, eRz, eFz = eps_zn
    rows = [
        ("Copper, 2000-2010 observed flow",
         ePc * cu["p_obs"], eRc * cu["r_obs"], eFc * cu["s_obs"]),
    ]
    if zn is not None:          # zinc row needs the (non-public) workbook
        rows.append(("Zinc, 1980-2019 observed flow",
                     ePz * zn["p_mean"], eRz * zn["r_mean"], eFz * zn["s_mean"]))
    df = pd.DataFrame(rows, columns=["Metal", "Primary", "Recycling", "Fabrication"])
    df["Total"] = df[["Primary", "Recycling", "Fabrication"]].sum(axis=1)
    return df
