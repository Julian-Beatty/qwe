from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def _ensure_dtindex(stock_df: pd.DataFrame, spot_col: str) -> pd.DataFrame:
    df = stock_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            raise ValueError("stock_df must have DatetimeIndex or 'date' column.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    if spot_col not in df.columns:
        raise KeyError(f"stock_df missing '{spot_col}', has {list(df.columns)}")
    return df

def build_obs_by_maturity(
    result_dict: Dict[str, dict],
    stock_df: pd.DataFrame,
    *,
    spot_col: str,
    align: str = "pad",
    max_gap_days: int = 3,
    min_horizon_days: int = 1,
) -> tuple[np.ndarray, List[List[dict]]]:
    """
    Returns:
      T_grid: (nT,)
      obs_by_T: list of length nT, each element is a list of dict obs with keys:
        S0, r, sigma, K_grid, qK_row, R_real, anchor_key, anchor_date
    """
    stock = _ensure_dtindex(stock_df, spot_col)

    keys = list(result_dict.keys())
    keys.sort()  # ensure chronological

    first = result_dict[keys[0]]
    T_grid = np.asarray(first["T_grid"], float).ravel()
    nT = T_grid.size

    def align_date(d: pd.Timestamp) -> Optional[pd.Timestamp]:
        idx = stock.index.get_indexer([d], method=align)
        pos = int(idx[0])
        if pos < 0:
            return None
        d0 = stock.index[pos]
        if abs((d0 - d).days) > int(max_gap_days):
            return None
        return d0

    obs_by_T: List[List[dict]] = [[] for _ in range(nT)]

    for k in keys:
        dd = result_dict[k]
        if not dd.get("success", True):
            continue
        if "atm_vol" not in dd or not np.isfinite(dd["atm_vol"]) or dd["atm_vol"] <= 0:
            continue

        anchor_date = pd.Timestamp(k).tz_localize(None)
        d0 = align_date(anchor_date)
        if d0 is None:
            continue

        S0 = float(dd.get("S0", np.nan))
        if not np.isfinite(S0) or S0 <= 0:
            S0 = float(stock.loc[d0, spot_col])
        if not np.isfinite(S0) or S0 <= 0:
            continue

        K_grid = np.asarray(dd["K_grid"], float).ravel()
        qK = np.asarray(dd["rnd_surface"], float)  # (nT, nK)
        if qK.ndim != 2 or qK.shape[0] != nT:
            continue

        r_rf = float(dd.get("r", 0.0))
        sigma = float(dd["atm_vol"])

        for j in range(nT):
            Tj = float(T_grid[j])
            if not np.isfinite(Tj) or Tj <= 0:
                continue
            horizon_days = int(np.round(Tj * 365.0))
            if horizon_days < min_horizon_days:
                continue

            end_date = d0 + pd.Timedelta(days=horizon_days)
            idx1 = stock.index.get_indexer([end_date], method="bfill")
            pos1 = int(idx1[0])
            if pos1 < 0:
                continue

            S1 = float(stock.iloc[pos1][spot_col])
            if not np.isfinite(S1) or S1 <= 0:
                continue

            R_real = float(S1 / S0)
            qK_row = np.asarray(qK[j, :], float)

            obs_by_T[j].append(
                dict(
                    anchor_key=str(k),
                    anchor_date=d0,
                    S0=S0,
                    r=r_rf,
                    sigma=sigma,
                    K_grid=K_grid,
                    qK_row=qK_row,
                    R_real=R_real,
                )
            )

    return T_grid, obs_by_T
