from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from rawinair.data import (
    AQ_VARS,
    MET_VARS,
    fit_scalers,
    load_knownair_bthsa,
    make_start_indices,
    year_mask,
)
from rawinair.graph import build_radius_graph_km
from rawinair.metrics import mae, per_target_metrics, rmse, smape
from rawinair.model import RawinAir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NC_PATH = PROJECT_ROOT / "data" / "BTHSAData" / "dataset_bthsa.nc"
DEFAULT_STATIONS_PATH = PROJECT_ROOT / "data" / "BTHSAData" / "stations_bthsa.csv"
DEFAULT_RUN_DIR = PROJECT_ROOT / "methods" / "RawinAir" / "runs" / "rawinair_bthsa"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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
    target_names: list[str] | None = None,
) -> dict[str, float]:
    model.eval()
    maes = []
    rmses = []
    smapes = []
    l1s_normalized = []  # 标准化空间的 L1（与 train 同尺度）
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

        # 标准化空间的 L1（与训练时一致）
        l1_norm = torch.mean(torch.abs(pred.float() - y.float()))
        l1s_normalized.append(float(l1_norm.cpu()))

        # de-normalize for metrics
        pred_den = (pred.float() * x_std.float()) + x_mean.float()
        y_den = (y.float() * x_std.float()) + x_mean.float()
        maes.append(float(mae(pred_den, y_den).cpu()))
        rmses.append(float(rmse(pred_den, y_den).cpu()))
        smapes.append(float(smape(pred_den, y_den).cpu()))
        per_list.append(per_target_metrics(pred_den, y_den, target_names=target_names))

    out: dict[str, float] = {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "smape": float(np.mean(smapes)),
        "l1_normalized": float(np.mean(l1s_normalized)),  # 同尺度 val L1
    }
    if per_list:
        keys = sorted(per_list[0].keys())
        for k in keys:
            out[k] = float(np.mean([d[k] for d in per_list]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=str(DEFAULT_NC_PATH))
    ap.add_argument("--stations", default=str(DEFAULT_STATIONS_PATH))
    ap.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))

    ap.add_argument("--t_hist", type=int, default=24)
    ap.add_argument("--t_pred", type=int, default=72)
    ap.add_argument("--radius_km", type=float, default=200.0)

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", help="训练设备，例如 cuda、cuda:0、cuda:1、cpu")
    ap.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["cosine", "plateau", "none"],
        help="学习率调度器：cosine=CosineAnnealingLR, plateau=ReduceLROnPlateau, none=不使用",
    )
    ap.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience（连续 N 个 epoch 无改善则停止，0=禁用）")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Early stopping 最小改善阈值（val_mae 需至少改善此值才算改善）")
    ap.add_argument("--max_train_batches", type=int, default=0, help=">0 则每个 epoch 仅跑前 N 个 batch（用于冒烟测试）")
    ap.add_argument("--max_val_batches", type=int, default=0, help=">0 则评估时仅跑前 N 个 batch（用于冒烟测试）")
    ap.add_argument("--amp", action="store_true", default=True, help="启用 CUDA 混合精度训练（torch.cuda.amp）")
    ap.add_argument(
        "--amp_dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP 精度类型：bf16 更稳（动态范围大），fp16 更快但更易溢出",
    )

    # split by year (matching paper)
    ap.add_argument("--train_years", default="2016,2017,2018,2019")
    ap.add_argument("--val_years", default="2020,2021")
    ap.add_argument("--test_years", default="2022,2023")

    # Ablations
    ap.add_argument("--disable_lid", action="store_true", help="消融：禁用 LID 模块")
    ap.add_argument("--disable_std", action="store_true", help="消融：禁用 STD 模块")
    ap.add_argument("--disable_tad", action="store_true", help="消融：禁用 TAD(GRU) 模块")
    ap.add_argument("--enable_dynamic_std_gate", action="store_true", help="启用风场引导的动态传播门控（作用于 STD 模块）")
    ap.add_argument("--dynamic_gate_hidden_dim", type=int, default=32, help="动态图门控内部维度")
    ap.add_argument("--dynamic_gate_temperature", type=float, default=1.0, help="动态图门控 softmax 温度")
    ap.add_argument("--dynamic_gate_blend_init", type=float, default=-3.0, help="动态图门控与静态 STD 融合的初始 raw gate")
    ap.add_argument("--enable_wind_direction_graph", action="store_true", help="启用风向感知传播图（作用于 STD 模块）")
    ap.add_argument("--wind_direction_blend_init", type=float, default=-2.0, help="风向传播图与现有 STD 融合的初始 raw gate")
    ap.add_argument("--wind_direction_scale_init", type=float, default=2.0, help="风向一致性映射到边权时的初始缩放系数")
    ap.add_argument("--enable_wind_history_gate", action="store_true", help="启用历史风速驱动的风传播边级可靠性门")
    ap.add_argument("--wind_history_gate_hidden_dim", type=int, default=16, help="历史风速可靠性门 MLP 隐层维度")
    ap.add_argument("--wind_history_gate_init_bias", type=float, default=2.0, help="历史风速可靠性门输出层 bias 初值")
    ap.add_argument("--wind_history_gate_residual_init", type=float, default=0.0, help="历史风场可靠性残差调制系数初值")
    ap.add_argument("--enable_light_dynamic_graph", action="store_true", help="启用轻量动态图分支（节点级低自由度边权调制）")
    ap.add_argument("--light_dynamic_hidden_dim", type=int, default=16, help="轻量动态图内部维度")
    ap.add_argument("--light_dynamic_blend_init", type=float, default=-3.0, help="轻量动态图与现有 STD 融合的初始 raw gate")
    ap.add_argument("--light_dynamic_scale_init", type=float, default=1.0, help="轻量动态图节点打分映射到边权时的初始缩放系数")
    ap.add_argument("--enable_adaptive_prior_embedding", action="store_true", help="启用时空先验注入（node adaptive embedding + time embedding）")
    ap.add_argument("--node_emb_dim", type=int, default=8, help="节点自适应嵌入维度")
    ap.add_argument("--time_emb_dim", type=int, default=8, help="时间步嵌入维度")
    ap.add_argument("--max_time_steps", type=int, default=256, help="时间步嵌入表最大长度")
    ap.add_argument("--adaptive_prior_blend_init", type=float, default=-2.0, help="时空先验注入到表示层的初始 raw gate")
    ap.add_argument("--enable_structured_prop_bias", action="store_true", help="启用传播边权结构化偏置（参考 logits bias 思路）")
    ap.add_argument("--structured_prop_dim", type=int, default=8, help="传播偏置低秩嵌入维度")
    ap.add_argument("--structured_prop_scale_init", type=float, default=1.0, help="结构化传播偏置初始缩放系数")
    ap.add_argument("--enable_dual_branch_fusion", action="store_true", help="启用结构分支与风向分支的双分支传播融合")
    ap.add_argument("--dual_branch_fusion_init", type=float, default=-2.0, help="双分支融合门的初始 raw gate")
    ap.add_argument("--enable_dual_branch_residual", action="store_true", help="启用双分支残差补充分支（物理主干之外的小动态图残差）")
    ap.add_argument("--residual_hidden_dim", type=int, default=16, help="残差分支内部维度")
    ap.add_argument("--residual_blend_init", type=float, default=-3.0, help="残差分支融合到物理主干的初始 raw gate")
    ap.add_argument("--residual_scale_init", type=float, default=1.0, help="残差分支边权打分的初始缩放系数")
    ap.add_argument(
        "--best_metric",
        default="pm25_rmse",
        choices=["pm25_rmse", "mae"],
        help="Checkpoint selection metric. Default uses PM2.5 RMSE to align with your main target.",
    )

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

    data = load_knownair_bthsa(args.nc, args.stations, aq_vars=AQ_VARS, met_vars=MET_VARS)

    train_years = set(int(x) for x in args.train_years.split(",") if x.strip())
    val_years = set(int(x) for x in args.val_years.split(",") if x.strip())
    test_years = set(int(x) for x in args.test_years.split(",") if x.strip())

    m_train = year_mask(data.time, train_years)
    m_val = year_mask(data.time, val_years)
    m_test = year_mask(data.time, test_years)

    # Fit scalers on training years only
    x_scaler, met_scaler = fit_scalers(data.x[m_train], data.met[m_train])

    # Build window starts that don't cross split boundaries
    starts_train = make_start_indices(m_train, args.t_hist, args.t_pred)
    starts_val = make_start_indices(m_val, args.t_hist, args.t_pred)
    starts_test = make_start_indices(m_test, args.t_hist, args.t_pred)

    from rawinair.data import KnownAirWindowDataset  # local import to keep file tidy

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
    
    # 详细图统计信息
    n_nodes = graph.num_nodes
    n_edges = graph.edge_index.shape[1]
    deg = torch.sparse.sum(graph.adj, dim=1).to_dense()
    n_self_loops = int((graph.edge_index[0] == graph.edge_index[1]).sum())
    n_isolated = int((deg == 0).sum())
    
    log(f"=== Graph Statistics ===")
    log(f"  nodes (N):          {n_nodes}")
    log(f"  directed edges (E): {n_edges}")
    log(f"  undirected edges:   ~{n_edges // 2} (symmetric)")
    log(f"  self-loops:         {n_self_loops}")
    log(f"  isolated nodes:     {n_isolated}")
    log(f"  degree (min/mean/max): {deg.min():.0f} / {deg.mean():.2f} / {deg.max():.0f}")
    log(f"  Ltilde shape:       {tuple(graph.ltilde_dense.shape)}")
    log(f"  Ltilde dtype:       {graph.ltilde_dense.dtype}")
    log(f"  Ltilde diag [0:5]:  {graph.ltilde_dense.diag()[:5].tolist()}")
    log(f"========================")

    # Attach graph to datasets for easy access in loader
    ds_train.graph = graph  # type: ignore[attr-defined]
    ds_val.graph = graph  # type: ignore[attr-defined]
    ds_test.graph = graph  # type: ignore[attr-defined]

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = RawinAir(
        in_dim=2,
        met_dim=len(MET_VARS),
        out_dim=2,
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

    # 学习率调度器
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
        log(f"Using CosineAnnealingLR scheduler (T_max={args.epochs}, eta_min={args.lr * 0.01:.2e})")
    elif args.scheduler == "plateau":
        # scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=args.lr * 0.01, verbose=True)
        scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=args.lr * 0.01)
        monitor_name = "PM2.5_rmse" if args.best_metric == "pm25_rmse" else "val_mae"
        log(f"Using ReduceLROnPlateau scheduler (monitor={monitor_name}, factor=0.5, patience=5)")
    else:
        scheduler = None
        log("No scheduler (constant learning rate)")

    # Early stopping
    early_stop_enabled = args.early_stop_patience > 0
    if early_stop_enabled:
        log(f"Early stopping enabled (patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta})")
    else:
        log("Early stopping disabled")

    x_mean = torch.tensor(x_scaler.mean, device=device, dtype=torch.float32).view(1, 1, 1, -1)
    x_std = torch.tensor(x_scaler.std, device=device, dtype=torch.float32).view(1, 1, 1, -1)

    best_val = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    ckpt_path = run_dir / "ckpt_best.pt"

    meta = {
        "args": vars(args),
        "aq_vars": AQ_VARS,
        "met_vars": MET_VARS,
        "station_ids": data.station_ids,
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
            # Forward under autocast for speed; compute losses in fp32 for stability.
            with autocast(enabled=bool(args.amp and device.type == "cuda"), dtype=amp_dtype):
                out = model(x_hist=x_hist, met_all=met_all, graph=graph, teacher_forcing=True)
                pred = out.y_pred
                transport = out.transport_pred

            pred_f = pred.float()
            y_f = y.float()

            l1 = torch.mean(torch.abs(pred_f - y_f))
            loss = l1

            if not torch.isfinite(loss):
                log(
                    f"WARN: non-finite loss detected (epoch={epoch} batch={bi}) "
                    f"loss={float(loss.detach().cpu())} l1={float(l1.detach().cpu())}"
                )
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
            target_names=AQ_VARS,
        )
        train_loss = float(np.mean(losses))
        val_l1_norm = val["l1_normalized"]  # 同尺度 val L1

        # 更新学习率调度器
        current_lr = opt.param_groups[0]["lr"]
        monitor_value = val["PM2.5_rmse"] if args.best_metric == "pm25_rmse" else val["mae"]
        if args.scheduler == "plateau":
            scheduler.step(monitor_value)
            new_lr = opt.param_groups[0]["lr"]
            if new_lr != current_lr:
                log(f"  -> LR reduced: {current_lr:.2e} -> {new_lr:.2e}")
        elif scheduler is not None:
            scheduler.step()
            new_lr = opt.param_groups[0]["lr"]
            if new_lr != current_lr:
                log(f"  -> LR updated: {current_lr:.2e} -> {new_lr:.2e}")

        t_epoch = time.perf_counter() - t_epoch0
        epoch_times.append(t_epoch)
        avg_epoch = float(np.mean(epoch_times))
        eta_s = avg_epoch * (args.epochs - epoch)
        log(
            f"epoch {epoch:03d}/{args.epochs} "
            f"time={t_epoch:.1f}s eta≈{eta_s/60:.1f}min lr={current_lr:.2e} "
            f"train_loss={train_loss:.6f} train_l1={np.mean(l1s):.6f} "
            f"val_l1={val_l1_norm:.6f} val_mae={val['mae']:.6f} val_rmse={val['rmse']:.6f} val_smape={val['smape']:.4f} "
            f"{AQ_VARS[0]}(mae/rmse/smape)={val[f'{AQ_VARS[0]}_mae']:.3f}/{val[f'{AQ_VARS[0]}_rmse']:.3f}/{val[f'{AQ_VARS[0]}_smape']:.4f} "
            f"{AQ_VARS[1]}(mae/rmse/smape)={val[f'{AQ_VARS[1]}_mae']:.3f}/{val[f'{AQ_VARS[1]}_rmse']:.3f}/{val[f'{AQ_VARS[1]}_smape']:.4f}"
        )

        # 检查是否改善并保存最佳模型
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

        # Early stopping
        if early_stop_enabled and early_stop_counter >= args.early_stop_patience:
            metric_label = "PM2.5_rmse" if args.best_metric == "pm25_rmse" else "val_mae"
            log(f"Early stopping triggered at epoch {epoch} (best: epoch {best_epoch}, {metric_label}={best_val:.6f})")
            break

    # Final test on best checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        metric_name = ckpt.get("best_metric_name", "val_mae")
        metric_value = ckpt.get("best_metric_value", ckpt.get("best_val_mae", float("nan")))
        log(f"Loaded best checkpoint from epoch {ckpt.get('best_epoch', 'unknown')} ({metric_name}={metric_value:.6f})")
    else:
        log("WARN: No checkpoint found, using final model")
    test = evaluate(
        model, dl_test, device, x_mean=x_mean, x_std=x_std, amp=bool(args.amp), amp_dtype=amp_dtype, target_names=AQ_VARS
    )
    t_total = time.perf_counter() - t0
    log(f"test: {test}")
    log(f"total_time: {t_total:.1f}s ({t_total/60:.1f}min)")
    (run_dir / "test_metrics.json").write_text(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
