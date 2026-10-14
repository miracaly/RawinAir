from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .data import KnownAirWindowDataset, StandardScaler, fit_scalers, make_start_indices, year_mask


PM25_INDEX = 17

ALL_FEATURE_NAMES = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "boundary_layer_height",
    "k_index",
    "relative_humidity+950",
    "relative_humidity+975",
    "specific_humidity+950",
    "surface_pressure",
    "temperature+925",
    "temperature+950",
    "total_precipitation",
    "u_component_of_wind+950",
    "v_component_of_wind+950",
    "vertical_velocity+950",
    "vorticity+950",
    "pm25",
]

STANDARD_MET_NAMES = [
    "2m_temperature",
    "boundary_layer_height",
    "k_index",
    "relative_humidity+950",
    "surface_pressure",
    "total_precipitation",
    "u_component_of_wind+950",
    "v_component_of_wind+950",
]

DERIVED_FEATURE_NAMES = ["hour", "weekday", "wind_speed", "wind_direction"]


@dataclass
class KnowAirCityData:
    station_ids: list[str]
    station_names: list[str]
    lon: np.ndarray
    lat: np.ndarray
    time: np.ndarray
    x: np.ndarray
    met: np.ndarray
    target_names: list[str]
    met_feature_names: list[str]


def _wind_indices(wind_source: Literal["950", "100m"]) -> tuple[int, int]:
    if wind_source == "950":
        return 13, 14
    return 0, 1


def _build_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Meteorological wind direction in degrees:
    direction FROM which the wind blows, clockwise from north.
    """
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def _base_feature_indices(
    met_feature_setup: Literal["uv100", "standard", "all17"],
    wind_source: Literal["950", "100m"],
) -> tuple[list[int], list[str]]:
    u_idx, v_idx = _wind_indices(wind_source)
    if met_feature_setup == "uv100":
        if wind_source != "100m":
            raise ValueError("met_feature_setup='uv100' requires wind_source='100m'.")
        ordered = [u_idx, v_idx]
        names = [ALL_FEATURE_NAMES[u_idx], ALL_FEATURE_NAMES[v_idx]]
        return ordered, names

    # Treat unrecognized values from older checkpoint metadata as the standard
    # feature set, while keeping "all17" as the only explicit full-set option.
    if met_feature_setup != "all17":
        ordered = [u_idx, v_idx, 3, 4, 5, 6, 9, 12]
        names = [
            ALL_FEATURE_NAMES[u_idx],
            ALL_FEATURE_NAMES[v_idx],
            "2m_temperature",
            "boundary_layer_height",
            "k_index",
            "relative_humidity+950",
            "surface_pressure",
            "total_precipitation",
        ]
        return ordered, names

    rest = [i for i in range(PM25_INDEX) if i not in {u_idx, v_idx}]
    ordered = [u_idx, v_idx] + rest
    names = [ALL_FEATURE_NAMES[i] for i in ordered]
    return ordered, names


def load_knowair_city(
    npy_path: str,
    city_txt_path: str,
    met_feature_setup: Literal["uv100", "standard", "all17"] = "standard",
    wind_source: Literal["950", "100m"] = "950",
    include_derived_features: bool = True,
) -> KnowAirCityData:
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 18:
        raise ValueError(f"Unexpected KnowAir.npy shape: {arr.shape}")

    city_path = Path(city_txt_path)
    rows = []
    for line in city_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed city.txt line: {line}")
        idx = int(parts[0])
        name = parts[1]
        lon = float(parts[2])
        lat = float(parts[3])
        rows.append((idx, name, lon, lat))
    rows.sort(key=lambda x: x[0])

    if len(rows) != arr.shape[1]:
        raise ValueError(f"city count {len(rows)} != node count {arr.shape[1]}")

    station_ids = [str(idx) for idx, _, _, _ in rows]
    station_names = [name for _, name, _, _ in rows]
    lon = np.asarray([x[2] for x in rows], dtype=np.float64)
    lat = np.asarray([x[3] for x in rows], dtype=np.float64)

    target = arr[:, :, PM25_INDEX : PM25_INDEX + 1].astype(np.float32)
    raw_features = arr[:, :, :PM25_INDEX].astype(np.float32)

    base_indices, base_names = _base_feature_indices(met_feature_setup, wind_source)
    met = raw_features[:, :, base_indices]
    met_feature_names = list(base_names)

    if include_derived_features:
        u = met[:, :, 0]
        v = met[:, :, 1]
        speed = np.sqrt(u * u + v * v).astype(np.float32)
        direction = _build_direction_deg(u, v).astype(np.float32)

        time_index = pd.date_range(
            start="2015-01-01 00:00:00+00:00",
            periods=arr.shape[0],
            freq="3H",
            tz="UTC",
        )
        hour = time_index.hour.to_numpy(dtype=np.float32)
        weekday = time_index.dayofweek.to_numpy(dtype=np.float32)
        hour = np.repeat(hour[:, None], arr.shape[1], axis=1)
        weekday = np.repeat(weekday[:, None], arr.shape[1], axis=1)

        derived = np.stack([hour, weekday, speed, direction], axis=-1).astype(np.float32)
        met = np.concatenate([met, derived], axis=-1)
        met_feature_names.extend(DERIVED_FEATURE_NAMES)

    time = pd.date_range(
        start="2015-01-01 00:00:00+00:00",
        periods=arr.shape[0],
        freq="3H",
        tz="UTC",
    ).to_numpy()

    return KnowAirCityData(
        station_ids=station_ids,
        station_names=station_names,
        lon=lon,
        lat=lat,
        time=time,
        x=target,
        met=met,
        target_names=["PM2.5"],
        met_feature_names=met_feature_names,
    )


__all__ = [
    "ALL_FEATURE_NAMES",
    "DERIVED_FEATURE_NAMES",
    "KnownAirWindowDataset",
    "KnowAirCityData",
    "STANDARD_MET_NAMES",
    "PM25_INDEX",
    "StandardScaler",
    "fit_scalers",
    "load_knowair_city",
    "make_start_indices",
    "year_mask",
]
