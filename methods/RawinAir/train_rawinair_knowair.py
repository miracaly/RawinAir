from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from rawinair.data_knowair import (
    KnownAirWindowDataset,
    fit_scalers,
    load_knowair_city,
    make_start_indices,
    year_mask,
)
from rawinair.graph import build_radius_graph_km
from rawinair.metrics import mae, per_target_metrics, rmse, smape
from rawinair.model import RawinAir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPY_PATH = PROJECT_ROOT / "data" / "KnowAirData" / "KnowAir.npy"
DEFAULT_CITY_PATH = PROJECT_ROOT / "data" / "KnowAirData" / "city.txt"
DEFAULT_RUN_DIR = PROJECT_ROOT / "methods" / "RawinAir" / "runs" / "rawinair_knowair"
TARGET_NAMES = ["PM2.5"]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@torch.no_grad()
def sequence_evaluation_metrics(
    x_hist_den: torch.Tensor,
    pred_den: torch.Tensor,
    y_den: torch.Tensor,
) -> dict[str, float]:
    """
    Match complete-sequence evaluation style:
    - concatenate history + future prediction
    - reshape to (samples, seq_len) across cities
    - MAE: mean over per-sample temporal MAE
    - RMSE: mean over per-sample temporal RMSE
    """
    predict_epoch = torch.cat([x_hist_den, pred_den], dim=1)
    label_epoch = torch.cat([x_hist_den, y_den], dim=1)
    predict = predict_epoch[..., 0].permute(0, 2, 1).reshape(-1, predict_epoch.shape[1])
    label = label_epoch[..., 0].permute(0, 2, 1).reshape(-1, label_epoch.shape[1])
    mae_v = torch.mean(torch.mean(torch.abs(predict - label), dim=1))
    rmse_v = torch.mean(torch.sqrt(torch.mean((predict - label) ** 2, dim=1)))
    return {
        "sequence_mae": float(mae_v.cpu()),
        "sequence_rmse": float(rmse_v.cpu()),
    }


@torch.no_grad()
def evaluate(
    model: RawinAir,
    loader: DataLoader,
    device: torch.device,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    max_batches: int | None = None,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    model.eval()
    maes = []
    rmses = []
    smapes = []
    l1s_normalized = []
    sequence_maes = []
    sequence_rmses = []
    per_list: list[dict[str, float]] = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x_hist = batch["x_hist"].to(device)
        met_all = batch["met_all"].to(device)
        y = batch["y"].to(device)
        with autocast(enabled=bool(amp and device.type == "cuda"), dtype=amp_dtype):
            out = model(x_hist=x_hist, met_all=met_all, graph=loader.dataset.graph)  # type: ignore[attr-defined]
            pred = out.y_pred

        l1_norm = torch.mean(torch.abs(pred.float() - y.float()))
        l1s_normalized.append(float(l1_norm.cpu()))

        pred_den = (pred.float() * x_std.float()) + x_mean.float()
        y_den = (y.float() * x_std.float()) + x_mean.float()
        x_hist_den = (x_hist.float() * x_std.float()) + x_mean.float()
        maes.append(float(mae(pred_den, y_den).cpu()))
        rmses.append(float(rmse(pred_den, y_den).cpu()))
        smapes.append(float(smape(pred_den, y_den).cpu()))
        sequence_metrics = sequence_evaluation_metrics(x_hist_den, pred_den, y_den)
        sequence_maes.append(sequence_metrics["sequence_mae"])
        sequence_rmses.append(sequence_metrics["sequence_rmse"])
        per_list.append(per_target_metrics(pred_den, y_den, target_names=TARGET_NAMES))

    out = {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "smape": float(np.mean(smapes)),
        "l1_normalized": float(np.mean(l1s_normalized)),
        "sequence_mae": float(np.mean(sequence_maes)),
        "sequence_rmse": float(np.mean(sequence_rmses)),
    }
    if per_list:
        for k in sorted(per_list[0].keys()):
            out[k] = float(np.mean([d[k] for d in per_list]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default=str(DEFAULT_NPY_PATH))
    ap.add_argument("--city", default=str(DEFAULT_CITY_PATH))
    ap.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--t_hist", type=int, default=24)
    ap.add_argument("--t_pred", type=int, default=24)
    ap.add_argument("--radius_km", type=float, default=300.0)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--scheduler", type=str, default="plateau", choices=["cosine", "plateau", "none"])
    ap.add_argument("--early_stop_patience", type=int, default=10)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0)
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_val_batches", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--train_years", default="2015,2016")
    ap.add_argument("--val_years", default="2017")
    ap.add_argument("--test_years", default="2018")
    ap.add_argument("--met_feature_setup", default="standard", choices=["uv100", "standard", "all17"])
    ap.add_argument("--wind_source", default="950", choices=["950", "100m"])
    ap.add_argument("--disable_derived_features", action="store_true")
    ap.add_argument("--disable_lid", action="store_true")
    ap.add_argument("--disable_std", action="store_true")
    ap.add_argument("--disable_tad", action="store_true")
    ap.add_argument("--enable_dynamic_std_gate", action="store_true")
    ap.add_argument("--dynamic_gate_hidden_dim", type=int, default=32)
    ap.add_argument("--dynamic_gate_temperature", type=float, default=1.0)
    ap.add_argument("--dynamic_gate_blend_init", type=float, default=-3.0)
    ap.add_argument("--enable_wind_direction_graph", action="store_true")
    ap.add_argument("--wind_direction_blend_init", type=float, default=-2.0)
    ap.add_argument("--wind_direction_scale_init", type=float, default=2.0)
    ap.add_argument("--transport_direction", default="current", choices=["current", "reverse"])
    ap.add_argument("--enable_wind_history_gate", action="store_true")
    ap.add_argument("--wind_history_gate_hidden_dim", type=int, default=16)
    ap.add_argument("--wind_history_gate_init_bias", type=float, default=2.0)
    ap.add_argument("--wind_history_gate_residual_init", type=float, default=0.0)
    ap.add_argument("--enable_light_dynamic_graph", action="store_true")
    ap.add_argument("--light_dynamic_hidden_dim", type=int, default=16)
    ap.add_argument("--light_dynamic_blend_init", type=float, default=-3.0)
    ap.add_argument("--light_dynamic_scale_init", type=float, default=1.0)
    ap.add_argument("--enable_adaptive_prior_embedding", action="store_true")
    ap.add_argument("--node_emb_dim", type=int, default=8)
    ap.add_argument("--time_emb_dim", type=int, default=8)
    ap.add_argument("--max_time_steps", type=int, default=512)
    ap.add_argument("--adaptive_prior_blend_init", type=float, default=-2.0)
    ap.add_argument("--enable_structured_prop_bias", action="store_true")
    ap.add_argument("--structured_prop_dim", type=int, default=8)
    ap.add_argument("--structured_prop_scale_init", type=float, default=1.0)
    ap.add_argument("--enable_dual_branch_fusion", action="store_true")
    ap.add_argument("--dual_branch_fusion_init", type=float, default=-2.0)
    ap.add_argument("--enable_dual_branch_residual", action="store_true")
    ap.add_argument("--residual_hidden_dim", type=int, default=16)
    ap.add_argument("--residual_blend_init", type=float, default=-3.0)
    ap.add_argument("--residual_scale_init", type=float, default=1.0)
    ap.add_argument("--best_metric", default="pm25_rmse", choices=["pm25_rmse", "mae"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    requested_device = str(args.device).strip().lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        log(f"WARN: requested device={args.device}, but CUDA is unavailable; falling back to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log(f"device: {device}")
    if args.amp and device.type != "cuda":
        log("WARN: --amp 仅在 CUDA 上生效；当前为 CPU，将忽略 AMP。")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_knowair_city(
        args.npy,
        args.city,
        met_feature_setup=args.met_feature_setup,
        wind_source=args.wind_source,
        include_derived_features=not args.disable_derived_features,
    )

    train_years = set(int(x) for x in args.train_years.split(",") if x.strip())
    val_years = set(int(x) for x in args.val_years.split(",") if x.strip())
    test_years = set(int(x) for x in args.test_years.split(",") if x.strip())

    m_train = year_mask(data.time, train_years)
    m_val = year_mask(data.time, val_years)
    m_test = year_mask(data.time, test_years)

    x_scaler, met_scaler = fit_scalers(data.x[m_train], data.met[m_train])
    starts_train = make_start_indices(m_train, args.t_hist, args.t_pred)
    starts_val = make_start_indices(m_val, args.t_hist, args.t_pred)
    starts_test = make_start_indices(m_test, args.t_hist, args.t_pred)

    ds_train = KnownAirWindowDataset(
        x=data.x,
        met=data.met,
        start_indices=starts_train,
        t_hist=args.t_hist,
        t_pred=args.t_pred,
        x_scaler=x_scaler,
        met_scaler=met_scaler,
    )
    ds_val = KnownAirWindowDataset(
        x=data.x,
        met=data.met,
        start_indices=starts_val,
        t_hist=args.t_hist,
        t_pred=args.t_pred,
        x_scaler=x_scaler,
        met_scaler=met_scaler,
    )
    ds_test = KnownAirWindowDataset(
        x=data.x,
        met=data.met,
        start_indices=starts_test,
        t_hist=args.t_hist,
        t_pred=args.t_pred,
        x_scaler=x_scaler,
        met_scaler=met_scaler,
    )

    graph = build_radius_graph_km(data.station_ids, data.lon, data.lat, radius_km=args.radius_km)
    deg = torch.sparse.sum(graph.adj, dim=1).to_dense()
    log("=== Graph Statistics ===")
    log(f"  nodes (N):          {graph.num_nodes}")
    log(f"  directed edges (E): {graph.edge_index.shape[1]}")
    log(f"  undirected edges:   ~{graph.edge_index.shape[1] // 2} (symmetric)")
    log(f"  self-loops:         {int((graph.edge_index[0] == graph.edge_index[1]).sum())}")
    log(f"  isolated nodes:     {int((deg == 0).sum())}")
    log(f"  degree (min/mean/max): {deg.min():.0f} / {deg.mean():.2f} / {deg.max():.0f}")
    log("========================")

    ds_train.graph = graph  # type: ignore[attr-defined]
    ds_val.graph = graph  # type: ignore[attr-defined]
    ds_test.graph = graph  # type: ignore[attr-defined]

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = RawinAir(
        in_dim=1,
        met_dim=int(data.met.shape[-1]),
        out_dim=1,
        num_nodes=graph.num_nodes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        enable_lid=not args.disable_lid,
        enable_std=not args.disable_std,
        enable_tad=not args.disable_tad,
        enable_dynamic_std_gate=bool(args.enable_dynamic_std_gate),
        dynamic_gate_hidden_dim=int(args.dynamic_gate_hidden_dim),
        dynamic_gate_temperature=float(args.dynamic_gate_temperature),
        dynamic_gate_blend_init=float(args.dynamic_gate_blend_init),
        enable_wind_direction_graph=bool(args.enable_wind_direction_graph),
        wind_direction_blend_init=float(args.wind_direction_blend_init),
        wind_direction_scale_init=float(args.wind_direction_scale_init),
        transport_direction=str(args.transport_direction),
        enable_wind_history_gate=bool(args.enable_wind_history_gate),
        wind_history_gate_hidden_dim=int(args.wind_history_gate_hidden_dim),
        wind_history_gate_init_bias=float(args.wind_history_gate_init_bias),
        wind_history_gate_residual_init=float(args.wind_history_gate_residual_init),
        enable_light_dynamic_graph=bool(args.enable_light_dynamic_graph),
        light_dynamic_hidden_dim=int(args.light_dynamic_hidden_dim),
        light_dynamic_blend_init=float(args.light_dynamic_blend_init),
        light_dynamic_scale_init=float(args.light_dynamic_scale_init),
        enable_adaptive_prior_embedding=bool(args.enable_adaptive_prior_embedding),
        node_emb_dim=int(args.node_emb_dim),
        time_emb_dim=int(args.time_emb_dim),
        max_time_steps=int(args.max_time_steps),
        adaptive_prior_blend_init=float(args.adaptive_prior_blend_init),
        enable_structured_prop_bias=bool(args.enable_structured_prop_bias),
        structured_prop_dim=int(args.structured_prop_dim),
        structured_prop_scale_init=float(args.structured_prop_scale_init),
        enable_dual_branch_fusion=bool(args.enable_dual_branch_fusion),
        dual_branch_fusion_init=float(args.dual_branch_fusion_init),
        enable_dual_branch_residual=bool(args.enable_dual_branch_residual),
        residual_hidden_dim=int(args.residual_hidden_dim),
        residual_blend_init=float(args.residual_blend_init),
        residual_scale_init=float(args.residual_scale_init),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
        log(f"Using CosineAnnealingLR scheduler (T_max={args.epochs}, eta_min={args.lr * 0.01:.2e})")
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=args.lr * 0.01)
        monitor_name = "PM2.5_rmse" if args.best_metric == "pm25_rmse" else "val_mae"
        log(f"Using ReduceLROnPlateau scheduler (monitor={monitor_name}, factor=0.5, patience=5)")
    else:
        scheduler = None
        log("No scheduler (constant learning rate)")

    early_stop_enabled = args.early_stop_patience > 0
    log(
        f"Early stopping enabled (patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta})"
        if early_stop_enabled
        else "Early stopping disabled"
    )

    x_mean = torch.tensor(x_scaler.mean, device=device, dtype=torch.float32).view(1, 1, 1, -1)
    x_std = torch.tensor(x_scaler.std, device=device, dtype=torch.float32).view(1, 1, 1, -1)
    best_val = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    ckpt_path = run_dir / "ckpt_best.pt"

    meta = {
        "args": vars(args),
        "target_names": TARGET_NAMES,
        "met_feature_names": data.met_feature_names,
        "station_ids": data.station_ids,
        "station_names": data.station_names,
        "scalers": {
            "x_mean": x_scaler.mean.tolist(),
            "x_std": x_scaler.std.tolist(),
            "met_mean": met_scaler.mean.tolist(),
            "met_std": met_scaler.std.tolist(),
        },
        "graph": {
            "radius_km": args.radius_km,
            "edge_index": graph.edge_index.cpu().numpy().tolist(),
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    log(f"train windows: {len(ds_train)}  val windows: {len(ds_val)}  test windows: {len(ds_test)}")
    log(f"graph edges (directed): {graph.edge_index.shape[1]}")

    t0 = time.perf_counter()
    epoch_times: list[float] = []
    for epoch in range(1, args.epochs + 1):
        t_epoch0 = time.perf_counter()
        model.train()
        losses = []
        l1s = []
        for bi, batch in enumerate(dl_train):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            x_hist = batch["x_hist"].to(device)
            met_all = batch["met_all"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(args.amp and device.type == "cuda"), dtype=amp_dtype):
                out = model(x_hist=x_hist, met_all=met_all, graph=graph, teacher_forcing=True)
                pred = out.y_pred
            pred_f = pred.float()
            y_f = y.float()
            l1 = torch.mean(torch.abs(pred_f - y_f))
            loss = l1
            if not torch.isfinite(loss):
                log(f"WARN: non-finite loss detected (epoch={epoch} batch={bi}) loss={float(loss.detach().cpu())}")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            l1s.append(float(l1.detach().cpu()))

        val = evaluate(
            model,
            dl_val,
            device,
            x_mean=x_mean,
            x_std=x_std,
            max_batches=(args.max_val_batches if args.max_val_batches else None),
            amp=bool(args.amp),
            amp_dtype=amp_dtype,
        )
        train_loss = float(np.mean(losses))
        val_l1_norm = val["l1_normalized"]
        current_lr = opt.param_groups[0]["lr"]
        monitor_value = val["PM2.5_rmse"] if args.best_metric == "pm25_rmse" else val["mae"]
        if args.scheduler == "plateau":
            scheduler.step(monitor_value)
        elif scheduler is not None:
            scheduler.step()

        t_epoch = time.perf_counter() - t_epoch0
        epoch_times.append(t_epoch)
        eta_s = float(np.mean(epoch_times)) * (args.epochs - epoch)
        log(
            f"epoch {epoch:03d}/{args.epochs} time={t_epoch:.1f}s eta≈{eta_s/60:.1f}min lr={current_lr:.2e} "
            f"train_loss={train_loss:.6f} train_l1={np.mean(l1s):.6f} "
            f"val_l1={val_l1_norm:.6f} val_mae={val['mae']:.6f} val_rmse={val['rmse']:.6f} val_smape={val['smape']:.4f} "
            f"PM2.5(mae/rmse/smape)={val['PM2.5_mae']:.3f}/{val['PM2.5_rmse']:.3f}/{val['PM2.5_smape']:.4f} "
            f"sequence(mae/rmse)={val['sequence_mae']:.3f}/{val['sequence_rmse']:.3f}"
        )

        improved = monitor_value < (best_val - args.early_stop_min_delta)
        if improved:
            best_val = monitor_value
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": meta,
                    "best_metric_name": ("PM2.5_rmse" if args.best_metric == "pm25_rmse" else "val_mae"),
                    "best_metric_value": best_val,
                    "best_val_mae": val["mae"],
                    "best_val_pm25_rmse": val["PM2.5_rmse"],
                    "best_epoch": best_epoch,
                },
                ckpt_path,
            )
            metric_label = "PM2.5_rmse" if args.best_metric == "pm25_rmse" else "val_mae"
            log(f"  -> Best model saved ({metric_label}={best_val:.6f})")
        else:
            early_stop_counter += 1
            if early_stop_enabled:
                log(f"  -> No improvement ({early_stop_counter}/{args.early_stop_patience})")

        if early_stop_enabled and early_stop_counter >= args.early_stop_patience:
            metric_label = "PM2.5_rmse" if args.best_metric == "pm25_rmse" else "val_mae"
            log(f"Early stopping triggered at epoch {epoch} (best: epoch {best_epoch}, {metric_label}={best_val:.6f})")
            break

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        metric_name = ckpt.get("best_metric_name", "val_mae")
        metric_value = ckpt.get("best_metric_value", ckpt.get("best_val_mae", float("nan")))
        log(f"Loaded best checkpoint from epoch {ckpt.get('best_epoch', 'unknown')} ({metric_name}={metric_value:.6f})")
    else:
        log("WARN: No checkpoint found, using final model")

    test = evaluate(model, dl_test, device, x_mean=x_mean, x_std=x_std, amp=bool(args.amp), amp_dtype=amp_dtype)
    t_total = time.perf_counter() - t0
    log(f"test: {test}")
    log(f"total_time: {t_total:.1f}s ({t_total/60:.1f}min)")
    (run_dir / "test_metrics.json").write_text(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
