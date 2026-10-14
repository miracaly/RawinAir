from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from rawinair.data import KnownAirWindowDataset, StandardScaler, make_start_indices, year_mask
from rawinair.data_knowair import load_knowair_city
from rawinair.graph import build_radius_graph_km
from rawinair.metrics import mae, per_horizon_metrics, per_horizon_per_target_metrics, per_target_metrics, rmse, smape
from rawinair.model import RawinAir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPY_PATH = PROJECT_ROOT / "data" / "KnowAirData" / "KnowAir.npy"
DEFAULT_CITY_PATH = PROJECT_ROOT / "data" / "KnowAirData" / "city.txt"
DEFAULT_CKPT_PATH = PROJECT_ROOT / "methods" / "RawinAir" / "runs" / "rawinair_knowair" / "ckpt_best.pt"
TARGET_NAMES = ["PM2.5"]


@torch.no_grad()
def sequence_evaluation_metrics(
    x_hist_den: torch.Tensor,
    pred_den: torch.Tensor,
    y_den: torch.Tensor,
) -> dict[str, float]:
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
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT_PATH))
    ap.add_argument("--npy", default=str(DEFAULT_NPY_PATH))
    ap.add_argument("--city", default=str(DEFAULT_CITY_PATH))
    ap.add_argument("--out", default="")
    ap.add_argument("--t_hist", type=int, default=24)
    ap.add_argument("--t_pred", type=int, default=24)
    ap.add_argument("--radius_km", type=float, default=300.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--test_years", default="2018")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--print_target_horizon_mae", action="store_true", default=True)
    args = ap.parse_args()

    requested_device = str(args.device).strip().lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARN: requested device={args.device}, but CUDA is unavailable; falling back to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print("device:", device)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})
    arg_meta = meta.get("args", {})

    x_scaler = StandardScaler(
        mean=np.array(meta["scalers"]["x_mean"], dtype=np.float32),
        std=np.array(meta["scalers"]["x_std"], dtype=np.float32),
    )
    met_scaler = StandardScaler(
        mean=np.array(meta["scalers"]["met_mean"], dtype=np.float32),
        std=np.array(meta["scalers"]["met_std"], dtype=np.float32),
    )

    data = load_knowair_city(
        args.npy,
        args.city,
        met_feature_setup=arg_meta.get("met_feature_setup", "standard"),
        wind_source=arg_meta.get("wind_source", "950"),
        include_derived_features=not bool(arg_meta.get("disable_derived_features", False)),
    )
    test_years = set(int(x) for x in args.test_years.split(",") if x.strip())
    m_test = year_mask(data.time, test_years)
    starts_test = make_start_indices(m_test, args.t_hist, args.t_pred)

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
    ds_test.graph = graph  # type: ignore[attr-defined]
    dl = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = RawinAir(
        in_dim=1,
        met_dim=len(data.met_feature_names),
        out_dim=1,
        num_nodes=graph.num_nodes,
        hidden_dim=int(arg_meta.get("hidden_dim", 64)),
        dropout=float(arg_meta.get("dropout", 0.1)),
        enable_lid=not bool(arg_meta.get("disable_lid", False)),
        enable_std=not bool(arg_meta.get("disable_std", False)),
        enable_tad=not bool(arg_meta.get("disable_tad", False)),
        enable_dynamic_std_gate=bool(arg_meta.get("enable_dynamic_std_gate", False)),
        dynamic_gate_hidden_dim=int(arg_meta.get("dynamic_gate_hidden_dim", 32)),
        dynamic_gate_temperature=float(arg_meta.get("dynamic_gate_temperature", 1.0)),
        dynamic_gate_blend_init=float(arg_meta.get("dynamic_gate_blend_init", -3.0)),
        enable_wind_direction_graph=bool(arg_meta.get("enable_wind_direction_graph", False)),
        wind_direction_blend_init=float(arg_meta.get("wind_direction_blend_init", -2.0)),
        wind_direction_scale_init=float(arg_meta.get("wind_direction_scale_init", 2.0)),
        transport_direction=str(arg_meta.get("transport_direction", "current")),
        enable_wind_history_gate=bool(arg_meta.get("enable_wind_history_gate", False)),
        wind_history_gate_hidden_dim=int(arg_meta.get("wind_history_gate_hidden_dim", 16)),
        wind_history_gate_init_bias=float(arg_meta.get("wind_history_gate_init_bias", 2.0)),
        wind_history_gate_residual_init=float(arg_meta.get("wind_history_gate_residual_init", 0.0)),
        enable_light_dynamic_graph=bool(arg_meta.get("enable_light_dynamic_graph", False)),
        light_dynamic_hidden_dim=int(arg_meta.get("light_dynamic_hidden_dim", 16)),
        light_dynamic_blend_init=float(arg_meta.get("light_dynamic_blend_init", -3.0)),
        light_dynamic_scale_init=float(arg_meta.get("light_dynamic_scale_init", 1.0)),
        enable_adaptive_prior_embedding=bool(arg_meta.get("enable_adaptive_prior_embedding", False)),
        node_emb_dim=int(arg_meta.get("node_emb_dim", 8)),
        time_emb_dim=int(arg_meta.get("time_emb_dim", 8)),
        max_time_steps=int(arg_meta.get("max_time_steps", 512)),
        adaptive_prior_blend_init=float(arg_meta.get("adaptive_prior_blend_init", -2.0)),
        enable_structured_prop_bias=bool(arg_meta.get("enable_structured_prop_bias", False)),
        structured_prop_dim=int(arg_meta.get("structured_prop_dim", 8)),
        structured_prop_scale_init=float(arg_meta.get("structured_prop_scale_init", 1.0)),
        enable_dual_branch_fusion=bool(arg_meta.get("enable_dual_branch_fusion", False)),
        dual_branch_fusion_init=float(arg_meta.get("dual_branch_fusion_init", -2.0)),
        enable_dual_branch_residual=bool(arg_meta.get("enable_dual_branch_residual", False)),
        residual_hidden_dim=int(arg_meta.get("residual_hidden_dim", 16)),
        residual_blend_init=float(arg_meta.get("residual_blend_init", -3.0)),
        residual_scale_init=float(arg_meta.get("residual_scale_init", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    maes = []
    rmses = []
    smapes = []
    mae_t_list = []
    rmse_t_list = []
    sequence_maes = []
    sequence_rmses = []
    per_list = []
    horizon_per_target_list = []

    x_mean = torch.tensor(x_scaler.mean, device=device, dtype=torch.float32).view(1, 1, 1, -1)
    x_std = torch.tensor(x_scaler.std, device=device, dtype=torch.float32).view(1, 1, 1, -1)

    for batch in dl:
        x_hist = batch["x_hist"].to(device)
        met_all = batch["met_all"].to(device)
        y = batch["y"].to(device)
        with autocast(autocast_device, enabled=bool(args.amp and device.type == "cuda"), dtype=amp_dtype):
            out = model(x_hist=x_hist, met_all=met_all, graph=graph, teacher_forcing=True)
            pred = out.y_pred

        pred_den = pred.float() * x_std.float() + x_mean.float()
        y_den = y.float() * x_std.float() + x_mean.float()
        x_hist_den = x_hist.float() * x_std.float() + x_mean.float()
        maes.append(float(mae(pred_den, y_den).cpu()))
        rmses.append(float(rmse(pred_den, y_den).cpu()))
        smapes.append(float(smape(pred_den, y_den).cpu()))
        sequence_metrics = sequence_evaluation_metrics(x_hist_den, pred_den, y_den)
        sequence_maes.append(sequence_metrics["sequence_mae"])
        sequence_rmses.append(sequence_metrics["sequence_rmse"])
        per_list.append(per_target_metrics(pred_den, y_den, target_names=TARGET_NAMES))
        ph = per_horizon_metrics(pred_den, y_den)
        mae_t_list.append(ph["mae_t"].cpu().numpy())
        rmse_t_list.append(ph["rmse_t"].cpu().numpy())
        hpt = per_horizon_per_target_metrics(pred_den, y_den, target_names=TARGET_NAMES)
        horizon_per_target_list.append({k: v.cpu().numpy() for k, v in hpt.items()})

    result = {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "smape": float(np.mean(smapes)),
        "sequence_mae": float(np.mean(sequence_maes)),
        "sequence_rmse": float(np.mean(sequence_rmses)),
        "mae_t": np.mean(np.stack(mae_t_list, axis=0), axis=0).tolist(),
        "rmse_t": np.mean(np.stack(rmse_t_list, axis=0), axis=0).tolist(),
        "num_windows": int(len(ds_test)),
    }
    if per_list:
        for k in sorted(per_list[0].keys()):
            result[k] = float(np.mean([d[k] for d in per_list]))
    if horizon_per_target_list:
        for k in sorted(horizon_per_target_list[0].keys()):
            result[k] = np.mean(np.stack([d[k] for d in horizon_per_target_list], axis=0), axis=0).tolist()
        for name in TARGET_NAMES:
            for metric_name in ["mae", "rmse", "smape"]:
                seq_key = f"{name}_{metric_name}_t"
                if seq_key in result:
                    arr = np.asarray(result[seq_key], dtype=float)
                    result[f"{name}_{metric_name}_avg"] = float(np.mean(arr))

    out_path = Path(args.out) if args.out else ckpt_path.with_name("eval_test.json")
    out_path.write_text(json.dumps(result, indent=2))
    print("saved:", str(out_path))
    print("result_overall:", {k: result[k] for k in ["mae", "rmse", "smape", "sequence_mae", "sequence_rmse", "num_windows"]})
    print("result_per_target:", {k: result[k] for k in ["PM2.5_mae", "PM2.5_rmse", "PM2.5_smape"] if k in result})
    if args.print_target_horizon_mae and "PM2.5_mae_t" in result:
        print("PM2.5_mae_t:", result["PM2.5_mae_t"])
    if args.print_target_horizon_mae and "PM2.5_smape_t" in result:
        print("PM2.5_smape_t:", result["PM2.5_smape_t"])
    if "PM2.5_smape_avg" in result:
        print(f"PM2.5 SMAPE avg{args.t_pred}: {result['PM2.5_smape_avg']}")


if __name__ == "__main__":
    main()
