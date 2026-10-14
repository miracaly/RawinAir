from __future__ import annotations

import torch


@torch.no_grad()
def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


@torch.no_grad()
def smape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric MAPE aligned with AirDDE/Air-DualODE.

    The value is reported as a ratio, not a percentage:
      mean(|pred - target| / ((|pred| + |target|) / 2 + eps)).
    """
    denom = (pred.abs() + target.abs()) / 2.0 + eps
    return ((pred - target).abs() / denom).mean()


@torch.no_grad()
def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Backward-compatible alias for the AirDDE-style SMAPE metric.
    """
    return smape(pred, target, eps=eps)


@torch.no_grad()
def per_target_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_names: list[str] | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    pred/target: (B, T, N, C)
    Returns per-target MAE/RMSE/SMAPE in a flat dict:
      { "PM2.5_mae": ..., "PM2.5_rmse": ..., "PM2.5_smape": ..., ... }
    """
    assert pred.shape == target.shape and pred.dim() == 4
    c = pred.shape[-1]
    if target_names is None:
        target_names = [f"y{i}" for i in range(c)]
    if len(target_names) != c:
        raise ValueError(f"target_names length {len(target_names)} != C {c}")

    out: dict[str, float] = {}
    for i, name in enumerate(target_names):
        p = pred[..., i]
        y = target[..., i]
        out[f"{name}_mae"] = float((p - y).abs().mean().cpu())
        out[f"{name}_rmse"] = float(torch.sqrt(torch.mean((p - y) ** 2)).cpu())
        out[f"{name}_smape"] = float(smape(p, y, eps=eps).cpu())
    return out

@torch.no_grad()
def per_horizon_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    pred/target: (B, T, N, C)
    Returns horizon-wise MAE/RMSE as (T,) tensors.
    """
    assert pred.shape == target.shape and pred.dim() == 4
    err = pred - target
    mae_t = err.abs().mean(dim=(0, 2, 3))
    rmse_t = torch.sqrt(torch.mean(err**2, dim=(0, 2, 3)))
    return {"mae_t": mae_t, "rmse_t": rmse_t}


@torch.no_grad()
def per_horizon_per_target_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_names: list[str] | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """
    pred/target: (B, T, N, C)
    Returns per-target horizon-wise metrics as (T,) tensors:
      { "AQI_mae_t": ..., "AQI_rmse_t": ..., "AQI_smape_t": ..., ... }
    """
    assert pred.shape == target.shape and pred.dim() == 4
    c = pred.shape[-1]
    if target_names is None:
        target_names = [f"y{i}" for i in range(c)]
    if len(target_names) != c:
        raise ValueError(f"target_names length {len(target_names)} != C {c}")

    out: dict[str, torch.Tensor] = {}
    for i, name in enumerate(target_names):
        p = pred[..., i]  # (B,T,N)
        y = target[..., i]
        err = p - y
        out[f"{name}_mae_t"] = err.abs().mean(dim=(0, 2))
        out[f"{name}_rmse_t"] = torch.sqrt(torch.mean(err**2, dim=(0, 2)))
        denom = (p.abs() + y.abs()) / 2.0 + eps
        out[f"{name}_smape_t"] = (err.abs() / denom).mean(dim=(0, 2))
    return out
