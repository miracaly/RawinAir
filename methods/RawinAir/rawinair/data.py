from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr


AQ_VARS = ["PM2.5", "O3"]
# 仅保留 u100、v100 作为协变量，其他气象变量不参与模型输入
MET_VARS = ["u100", "v100"]


@dataclass
class StandardScaler:
    mean: np.ndarray  # (C,)
    std: np.ndarray  # (C,)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class KnownAirData:
    station_ids: list[str]
    lon: np.ndarray  # (N,)
    lat: np.ndarray  # (N,)
    time: np.ndarray  # (T,) datetime64[ns]
    x: np.ndarray  # (T,N,2) float32
    met: np.ndarray  # (T,N,8) float32


def load_knownair_bthsa(
    nc_path: str,
    stations_csv_path: str,
    aq_vars: Iterable[str] = AQ_VARS,
    met_vars: Iterable[str] = MET_VARS,
) -> KnownAirData:
    ds = xr.open_dataset(nc_path)
    station_ids = [str(s) for s in ds["station"].values.tolist()]

    df = pd.read_csv(stations_csv_path)
    df["station_id"] = df["station_id"].astype(str)
    meta = df.set_index("station_id")

    # Reorder lon/lat to match nc station order
    lon = meta.loc[station_ids]["lon"].to_numpy(dtype=np.float64)
    lat = meta.loc[station_ids]["lat"].to_numpy(dtype=np.float64)

    time = ds["time"].values

    aq_vars = list(aq_vars)
    met_vars = list(met_vars)
    x = np.stack([ds[v].values for v in aq_vars], axis=-1).astype(np.float32)  # (T,N,2)
    met = np.stack([ds[v].values for v in met_vars], axis=-1).astype(np.float32)  # (T,N,8)

    return KnownAirData(
        station_ids=station_ids,
        lon=lon,
        lat=lat,
        time=time,
        x=x,
        met=met,
    )


def year_mask(time: np.ndarray, years: set[int]) -> np.ndarray:
    # time: datetime64[ns]
    y = pd.DatetimeIndex(time).year.to_numpy()
    mask = np.isin(y, np.array(sorted(list(years)), dtype=np.int32))
    return mask


def fit_scalers(train_x: np.ndarray, train_met: np.ndarray) -> tuple[StandardScaler, StandardScaler]:
    # train_x: (T,N,Cx), train_met: (T,N,Cm)
    cx = train_x.shape[-1]
    cm = train_met.shape[-1]
    x_flat = train_x.reshape(-1, cx)
    m_flat = train_met.reshape(-1, cm)
    x_mean = x_flat.mean(axis=0)
    x_std = x_flat.std(axis=0) + 1e-6
    m_mean = m_flat.mean(axis=0)
    m_std = m_flat.std(axis=0) + 1e-6
    return StandardScaler(x_mean, x_std), StandardScaler(m_mean, m_std)


class KnownAirWindowDataset(Dataset):
    """
    Sliding window dataset.

    We use future covariates (meteorology) as known inputs.

    Returns:
      x_hist: (T_hist, N, 2)
      met_all: (T_hist + T_pred, N, 8)
      y: (T_pred, N, 2)
    """

    def __init__(
        self,
        x: np.ndarray,
        met: np.ndarray,
        start_indices: np.ndarray,
        time: Optional[np.ndarray] = None,
        t_hist: int = 24,
        t_pred: int = 72,
        x_scaler: Optional[StandardScaler] = None,
        met_scaler: Optional[StandardScaler] = None,
        add_time_features: bool = False,
    ):
        assert x.shape[0] == met.shape[0]
        self.x = x
        self.met = met
        self.time = time
        self.start_indices = start_indices.astype(np.int64)
        self.t_hist = int(t_hist)
        self.t_pred = int(t_pred)
        self.x_scaler = x_scaler
        self.met_scaler = met_scaler
        self.add_time_features = bool(add_time_features)
        self._base_met_dim = int(met.shape[-1])
        self.time_feat_dim = 0
        if self.add_time_features:
            if self.time is None:
                raise ValueError("add_time_features=True requires passing `time` to KnownAirWindowDataset.")
            # hour_sin/cos + doy_sin/cos + dow_sin/cos
            self.time_feat_dim = 6
        self.met_dim = self._base_met_dim + self.time_feat_dim

    def __len__(self) -> int:
        return int(self.start_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = int(self.start_indices[idx])
        th = self.t_hist
        tp = self.t_pred
        x_win = self.x[s : s + th + tp]  # (th+tp, N, 2)
        met_win = self.met[s : s + th + tp]  # (th+tp, N, 8)

        x_hist = x_win[:th]
        y = x_win[th:]

        if self.x_scaler is not None:
            x_hist = self.x_scaler.transform(x_hist)
            y = self.x_scaler.transform(y)

        # Add deterministic time features (known for future) before scaling.
        # Keep them in [-1, 1] range (sin/cos), do NOT scale with met_scaler.
        if self.add_time_features:
            assert self.time is not None
            t_win = self.time[s : s + th + tp]
            dt = pd.DatetimeIndex(t_win)
            hour = dt.hour.to_numpy(dtype=np.float32)  # 0..23
            doy = dt.dayofyear.to_numpy(dtype=np.float32)  # 1..366
            dow = dt.dayofweek.to_numpy(dtype=np.float32)  # 0..6

            # map to radians
            hour_rad = 2.0 * np.pi * (hour / 24.0)
            doy_rad = 2.0 * np.pi * ((doy - 1.0) / 366.0)
            dow_rad = 2.0 * np.pi * (dow / 7.0)

            feats = np.stack(
                [
                    np.sin(hour_rad),
                    np.cos(hour_rad),
                    np.sin(doy_rad),
                    np.cos(doy_rad),
                    np.sin(dow_rad),
                    np.cos(dow_rad),
                ],
                axis=-1,
            ).astype(np.float32)  # (w, 6)
            # broadcast across stations: (w, 1, 6) -> (w, N, 6)
            feats = feats[:, None, :]
            feats = np.repeat(feats, met_win.shape[1], axis=1)
            met_win = np.concatenate([met_win, feats], axis=-1)  # (w, N, base_met_dim + 6)

        if self.met_scaler is not None:
            # Scale only the original meteorology channels, keep time features as-is.
            met_scaled = self.met_scaler.transform(met_win[..., : self._base_met_dim])
            if self.time_feat_dim > 0:
                met_win = np.concatenate([met_scaled, met_win[..., self._base_met_dim :]], axis=-1)
            else:
                met_win = met_scaled

        return {
            "x_hist": torch.from_numpy(x_hist),
            "met_all": torch.from_numpy(met_win),
            "y": torch.from_numpy(y),
        }


def make_start_indices(mask: np.ndarray, t_hist: int, t_pred: int) -> np.ndarray:
    """
    Build valid window starts given a boolean mask (time axis).
    Window is valid if all timesteps within [s, s+t_hist+t_pred) are inside mask.
    """
    mask = mask.astype(bool)
    t = mask.shape[0]
    w = t_hist + t_pred
    # cumulative sum for fast range query
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int32))])
    starts = []
    for s in range(0, t - w + 1):
        inside = c[s + w] - c[s]
        if inside == w:
            starts.append(s)
    return np.array(starts, dtype=np.int64)


