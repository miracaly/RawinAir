from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance (km) between two sets of points.
    lon/lat in degrees.
    """
    r = 6371.0
    lon1r = np.deg2rad(lon1)
    lat1r = np.deg2rad(lat1)
    lon2r = np.deg2rad(lon2)
    lat2r = np.deg2rad(lat2)

    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


@dataclass(frozen=True)
class GraphData:
    """
    Sparse graph utilities for station-level message passing.

    We keep:
    - adjacency (A) as a sparse COO tensor (no self-loops by default)
    - normalized laplacian operator Ltilde = I - D^{-1/2} A D^{-1/2}
    """

    num_nodes: int
    edge_index: torch.Tensor  # (2, E) long, edges i->j (undirected stored as both directions)
    adj: torch.Tensor  # sparse_coo (N, N)
    adj_dense: torch.Tensor  # dense (N, N)
    ltilde: torch.Tensor  # sparse_coo (N, N)
    ltilde_dense: torch.Tensor  # dense (N, N), for fast batch matmul on GPU
    edge_dir_dense: torch.Tensor  # dense (N, N, 2), unit direction vector i->j in local lon/lat plane


def build_radius_graph_km(
    station_ids: list[str],
    lon: np.ndarray,
    lat: np.ndarray,
    radius_km: float = 200.0,
    add_self_loops: bool = False,
) -> GraphData:
    """
    Build an undirected radius graph using geodesic distance threshold.
    Returns edges in both directions.
    """
    n = len(station_ids)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    assert lon.shape == (n,) and lat.shape == (n,)

    # Some datasets may contain invalid/missing coordinates (e.g., -999).
    valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)

    # O(N^2) is OK for N=228.
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        if not valid[i]:
            continue
        di = haversine_km(lon[i], lat[i], lon, lat)
        js = np.where((di <= radius_km) & (di > 0) & valid)[0]
        for j in js.tolist():
            rows.append(i)
            cols.append(j)

    if add_self_loops:
        for i in range(n):
            rows.append(i)
            cols.append(i)
    elif not np.all(valid):
        # Ensure isolated nodes with invalid coords at least have a self-loop,
        # otherwise they would have zero degree and behave poorly.
        for i in np.where(~valid)[0].tolist():
            rows.append(i)
            cols.append(i)

    if len(rows) == 0:
        raise ValueError(f"No edges found with radius_km={radius_km}.")

    edge_index = torch.tensor([rows, cols], dtype=torch.long)

    values = torch.ones(edge_index.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, values, size=(n, n)).coalesce()
    adj_dense = adj.to_dense()

    edge_dir_dense = torch.zeros((n, n, 2), dtype=torch.float32)
    for i, j in zip(rows, cols):
        if i == j:
            continue
        mean_lat = math.radians(float(lat[i] + lat[j]) * 0.5)
        dx = float(lon[j] - lon[i]) * math.cos(mean_lat)
        dy = float(lat[j] - lat[i])
        norm = math.sqrt(dx * dx + dy * dy)
        if norm <= 1e-12:
            continue
        edge_dir_dense[i, j, 0] = dx / norm
        edge_dir_dense[i, j, 1] = dy / norm

    # Compute normalized Laplacian: Ltilde = I - D^{-1/2} A D^{-1/2}
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1.0)  # (N,)
    deg_inv_sqrt = deg.pow(-0.5)

    idx = adj.indices()
    v = adj.values()
    di = deg_inv_sqrt[idx[0]]
    dj = deg_inv_sqrt[idx[1]]
    norm_vals = v * di * dj
    norm_adj = torch.sparse_coo_tensor(idx, norm_vals, size=(n, n)).coalesce()

    eye_idx = torch.arange(n, dtype=torch.long)
    eye = torch.sparse_coo_tensor(
        torch.stack([eye_idx, eye_idx], dim=0),
        torch.ones(n, dtype=torch.float32),
        size=(n, n),
    ).coalesce()
    ltilde = (eye - norm_adj).coalesce()

    # For N=228, densifying is cheap and usually much faster than doing B times sparse.mm.
    ltilde_dense = ltilde.to_dense()

    return GraphData(
        num_nodes=n,
        edge_index=edge_index,
        adj=adj,
        adj_dense=adj_dense,
        ltilde=ltilde,
        ltilde_dense=ltilde_dense,
        edge_dir_dense=edge_dir_dense,
    )


def sparse_mm(sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    """
    Multiply sparse (N,N) with dense (N,H) or (B,N,H).
    """
    if dense.dim() == 2:
        return torch.sparse.mm(sparse, dense)
    if dense.dim() == 3:
        b, n, h = dense.shape
        out = []
        for i in range(b):
            out.append(torch.sparse.mm(sparse, dense[i]))
        return torch.stack(out, dim=0)
    raise ValueError(f"Unsupported dense shape: {dense.shape}")

