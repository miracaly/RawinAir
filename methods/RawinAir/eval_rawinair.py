from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from rawinair.data import (
    AQ_VARS,
    MET_VARS,
    KnownAirWindowDataset,
    StandardScaler,
    load_knownair_bthsa,
    make_start_indices,
    year_mask,
)
from rawinair.graph import build_radius_graph_km
from rawinair.metrics import mae, per_horizon_metrics, per_horizon_per_target_metrics, per_target_metrics, rmse, smape
from rawinair.model import RawinAir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NC_PATH = PROJECT_ROOT / "data" / "BTHSAData" / "dataset_bthsa.nc"
DEFAULT_STATIONS_PATH = PROJECT_ROOT / "data" / "BTHSAData" / "stations_bthsa.csv"
DEFAULT_CKPT_PATH = PROJECT_ROOT / "methods" / "RawinAir" / "runs" / "rawinair_bthsa" / "ckpt_best.pt"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT_PATH))
    ap.add_argument("--nc", default=str(DEFAULT_NC_PATH))
    ap.add_argument("--stations", default=str(DEFAULT_STATIONS_PATH))
    ap.add_argument("--out", default="", help="Output json path (default next to checkpoint)")
    ap.add_argument("--t_hist", type=int, default=24)
    ap.add_argument("--t_pred", type=int, default=72)
    ap.add_argument("--radius_km", type=float, default=200.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--test_years", default="2022,2023")
    ap.add_argument("--device", type=str, default="cuda", help="评估设备，例如 cuda、cuda:0、cuda:1、cpu")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])

    # per-horizon list printing (default: first target)
    ap.add_argument("--print_target_horizon_mae", action="store_true", default=True, help="输出指定目标的 per-horizon MAE list")
    ap.add_argument("--print_target", default="", help="要打印 per-horizon MAE 的目标名（默认第一个）")

    # plotting (include hist)
    ap.add_argument("--plot", action="store_true", help="输出一张真实值 vs 预测值曲线图（含hist）")
    ap.add_argument("--plot_station", default="", help="要画图的 station_id（默认第一个站点）")
    ap.add_argument("--plot_target", default="", help="要画图的目标名（默认第一个）")
    ap.add_argument("--plot_out", default="", help="图像输出路径（默认与 eval_test.json 同名 .plot.png）")

    # random plots (deterministic)
    ap.add_argument("--plot_random", action="store_true", help="随机站点/随机窗口画曲线（含hist），seed 固定")
    ap.add_argument("--plot_seed", type=int, default=42)
    ap.add_argument("--plot_n_windows", type=int, default=3)
    ap.add_argument("--plot_n_stations", type=int, default=3)
    ap.add_argument("--plot_dir", default="", help="随机绘图输出目录（默认 runs/ 同目录下 random_plots/）")
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

    # scalers from checkpoint
    s = meta.get("scalers", {})
    x_scaler = StandardScaler(mean=np.array(s["x_mean"], dtype=np.float32), std=np.array(s["x_std"], dtype=np.float32))
    met_scaler = StandardScaler(
        mean=np.array(s["met_mean"], dtype=np.float32), std=np.array(s["met_std"], dtype=np.float32)
    )

    data = load_knownair_bthsa(args.nc, args.stations, aq_vars=AQ_VARS, met_vars=MET_VARS)
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

    # model params: infer from checkpoint meta if present
    hidden_dim = int(meta.get("args", {}).get("hidden_dim", 64))
    dropout = float(meta.get("args", {}).get("dropout", 0.1))
    disable_lid = bool(meta.get("args", {}).get("disable_lid", False))
    disable_std = bool(meta.get("args", {}).get("disable_std", False))
    disable_tad = bool(meta.get("args", {}).get("disable_tad", False))
    enable_dynamic_std_gate = bool(meta.get("args", {}).get("enable_dynamic_std_gate", False))
    dynamic_gate_hidden_dim = int(meta.get("args", {}).get("dynamic_gate_hidden_dim", 32))
    dynamic_gate_temperature = float(meta.get("args", {}).get("dynamic_gate_temperature", 1.0))
    dynamic_gate_blend_init = float(meta.get("args", {}).get("dynamic_gate_blend_init", -3.0))
    enable_wind_direction_graph = bool(meta.get("args", {}).get("enable_wind_direction_graph", False))
    wind_direction_blend_init = float(meta.get("args", {}).get("wind_direction_blend_init", -2.0))
    wind_direction_scale_init = float(meta.get("args", {}).get("wind_direction_scale_init", 2.0))
    enable_wind_history_gate = bool(meta.get("args", {}).get("enable_wind_history_gate", False))
    wind_history_gate_hidden_dim = int(meta.get("args", {}).get("wind_history_gate_hidden_dim", 16))
    wind_history_gate_init_bias = float(meta.get("args", {}).get("wind_history_gate_init_bias", 2.0))
    wind_history_gate_residual_init = float(meta.get("args", {}).get("wind_history_gate_residual_init", 0.0))
    enable_light_dynamic_graph = bool(meta.get("args", {}).get("enable_light_dynamic_graph", False))
    light_dynamic_hidden_dim = int(meta.get("args", {}).get("light_dynamic_hidden_dim", 16))
    light_dynamic_blend_init = float(meta.get("args", {}).get("light_dynamic_blend_init", -3.0))
    light_dynamic_scale_init = float(meta.get("args", {}).get("light_dynamic_scale_init", 1.0))
    enable_adaptive_prior_embedding = bool(meta.get("args", {}).get("enable_adaptive_prior_embedding", False))
    node_emb_dim = int(meta.get("args", {}).get("node_emb_dim", 8))
    time_emb_dim = int(meta.get("args", {}).get("time_emb_dim", 8))
    max_time_steps = int(meta.get("args", {}).get("max_time_steps", 256))
    adaptive_prior_blend_init = float(meta.get("args", {}).get("adaptive_prior_blend_init", -2.0))
    enable_structured_prop_bias = bool(meta.get("args", {}).get("enable_structured_prop_bias", False))
    structured_prop_dim = int(meta.get("args", {}).get("structured_prop_dim", 8))
    structured_prop_scale_init = float(meta.get("args", {}).get("structured_prop_scale_init", 1.0))
    enable_dual_branch_fusion = bool(meta.get("args", {}).get("enable_dual_branch_fusion", False))
    dual_branch_fusion_init = float(meta.get("args", {}).get("dual_branch_fusion_init", -2.0))
    enable_dual_branch_residual = bool(meta.get("args", {}).get("enable_dual_branch_residual", False))
    residual_hidden_dim = int(meta.get("args", {}).get("residual_hidden_dim", 16))
    residual_blend_init = float(meta.get("args", {}).get("residual_blend_init", -3.0))
    residual_scale_init = float(meta.get("args", {}).get("residual_scale_init", 1.0))
    model = RawinAir(
        in_dim=2,
        met_dim=len(MET_VARS),
        out_dim=2,
        num_nodes=graph.num_nodes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        enable_lid=not disable_lid,
        enable_std=not disable_std,
        enable_tad=not disable_tad,
        enable_dynamic_std_gate=enable_dynamic_std_gate,
        dynamic_gate_hidden_dim=dynamic_gate_hidden_dim,
        dynamic_gate_temperature=dynamic_gate_temperature,
        dynamic_gate_blend_init=dynamic_gate_blend_init,
        enable_wind_direction_graph=enable_wind_direction_graph,
        wind_direction_blend_init=wind_direction_blend_init,
        wind_direction_scale_init=wind_direction_scale_init,
        enable_wind_history_gate=enable_wind_history_gate,
        wind_history_gate_hidden_dim=wind_history_gate_hidden_dim,
        wind_history_gate_init_bias=wind_history_gate_init_bias,
        wind_history_gate_residual_init=wind_history_gate_residual_init,
        enable_light_dynamic_graph=enable_light_dynamic_graph,
        light_dynamic_hidden_dim=light_dynamic_hidden_dim,
        light_dynamic_blend_init=light_dynamic_blend_init,
        light_dynamic_scale_init=light_dynamic_scale_init,
        enable_adaptive_prior_embedding=enable_adaptive_prior_embedding,
        node_emb_dim=node_emb_dim,
        time_emb_dim=time_emb_dim,
        max_time_steps=max_time_steps,
        adaptive_prior_blend_init=adaptive_prior_blend_init,
        enable_structured_prop_bias=enable_structured_prop_bias,
        structured_prop_dim=structured_prop_dim,
        structured_prop_scale_init=structured_prop_scale_init,
        enable_dual_branch_fusion=enable_dual_branch_fusion,
        dual_branch_fusion_init=dual_branch_fusion_init,
        enable_dual_branch_residual=enable_dual_branch_residual,
        residual_hidden_dim=residual_hidden_dim,
        residual_blend_init=residual_blend_init,
        residual_scale_init=residual_scale_init,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    maes = []
    rmses = []
    smapes = []
    mae_t_list = []
    rmse_t_list = []
    per_list = []
    horizon_per_target_list = []

    # pick defaults for target/station
    plot_target = args.plot_target or AQ_VARS[0]
    if plot_target not in AQ_VARS:
        plot_target = AQ_VARS[0]
    print_target = args.print_target or AQ_VARS[0]
    if print_target not in AQ_VARS:
        print_target = AQ_VARS[0]

    plot_data = None
    plot_station = args.plot_station or (data.station_ids[0] if data.station_ids else "")
    plot_station_idx = data.station_ids.index(plot_station) if plot_station in data.station_ids else 0

    for batch in dl:
        x_hist = batch["x_hist"].to(device)
        met_all = batch["met_all"].to(device)
        y = batch["y"].to(device)
        with autocast(autocast_device, enabled=bool(args.amp and device.type == "cuda"), dtype=amp_dtype):
            # align with training evaluation: teacher_forcing=True on history
            out = model(x_hist=x_hist, met_all=met_all, graph=graph, teacher_forcing=True)
            pred = out.y_pred
        # de-normalize for metrics
        x_mean = torch.tensor(x_scaler.mean, device=device, dtype=pred.dtype).view(1, 1, 1, -1)
        x_std = torch.tensor(x_scaler.std, device=device, dtype=pred.dtype).view(1, 1, 1, -1)
        pred_den = pred.float() * x_std.float() + x_mean.float()
        y_den = y.float() * x_std.float() + x_mean.float()

        maes.append(float(mae(pred_den, y_den).cpu()))
        rmses.append(float(rmse(pred_den, y_den).cpu()))
        smapes.append(float(smape(pred_den, y_den).cpu()))
        per_list.append(per_target_metrics(pred_den, y_den, target_names=AQ_VARS))
        ph = per_horizon_metrics(pred_den, y_den)
        mae_t_list.append(ph["mae_t"].cpu().numpy())
        rmse_t_list.append(ph["rmse_t"].cpu().numpy())

        hpt = per_horizon_per_target_metrics(pred_den, y_den, target_names=AQ_VARS)
        horizon_per_target_list.append({k: v.cpu().numpy() for k, v in hpt.items()})

        if args.plot and plot_data is None:
            b0 = 0
            target_i = AQ_VARS.index(plot_target)
            x_hist_den = x_hist.float() * x_std.float() + x_mean.float()
            hist_true = x_hist_den[b0, :, plot_station_idx, target_i].detach().cpu().numpy()
            fut_true = y_den[b0, :, plot_station_idx, target_i].detach().cpu().numpy()
            fut_pred = pred_den[b0, :, plot_station_idx, target_i].detach().cpu().numpy()
            plot_data = {
                "station_id": data.station_ids[plot_station_idx] if data.station_ids else "",
                "target": plot_target,
                "hist_true": hist_true,
                "y_true": fut_true,
                "y_pred": fut_pred,
            }

    result = {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "smape": float(np.mean(smapes)),
        "mae_t": np.mean(np.stack(mae_t_list, axis=0), axis=0).tolist(),
        "rmse_t": np.mean(np.stack(rmse_t_list, axis=0), axis=0).tolist(),
        "num_windows": int(len(ds_test)),
    }
    if per_list:
        keys = sorted(per_list[0].keys())
        for k in keys:
            result[k] = float(np.mean([d[k] for d in per_list]))
    if horizon_per_target_list:
        keys = sorted(horizon_per_target_list[0].keys())
        for k in keys:
            result[k] = np.mean(np.stack([d[k] for d in horizon_per_target_list], axis=0), axis=0).tolist()
    out_path = Path(args.out) if args.out else ckpt_path.with_name("eval_test.json")
    out_path.write_text(json.dumps(result, indent=2))
    print("saved:", str(out_path))
    print("result_overall:", {k: result[k] for k in ["mae", "rmse", "smape", "num_windows"]})

    per_overall = {}
    for name in AQ_VARS:
        for m in ["mae", "rmse", "smape"]:
            k = f"{name}_{m}"
            if k in result:
                per_overall[k] = result[k]
    print("result_per_target:", per_overall)

    if args.print_target_horizon_mae and f"{print_target}_mae_t" in result:
        print(f"{print_target}_mae_t:", result[f"{print_target}_mae_t"])

    # plot single curve
    if args.plot and plot_data is not None:
        try:
            import matplotlib.pyplot as plt

            hist = plot_data["hist_true"]
            fut_true = plot_data["y_true"]
            fut_pred = plot_data["y_pred"]
            t = np.arange(1, len(hist) + len(fut_true) + 1)
            true_series = np.concatenate([hist, fut_true], axis=0)
            pred_series = np.concatenate([hist, fut_pred], axis=0)
            plt.figure(figsize=(10, 4))
            plt.plot(t, true_series, label="true", linewidth=2)
            plt.plot(t, pred_series, label="pred", linewidth=2)
            plt.axvline(len(hist), color="k", linestyle="--", alpha=0.4, label="hist/pred split")
            plt.title(f"{plot_data['target']} true vs pred @ station {plot_data['station_id']}")
            plt.xlabel("step (hist+pred)")
            plt.ylabel(plot_data["target"])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_out = Path(args.plot_out) if args.plot_out else out_path.with_suffix(".plot.png")
            plt.tight_layout()
            plt.savefig(plot_out, dpi=160)
            plt.close()
            print("plot saved:", str(plot_out))
        except Exception as e:
            print("WARN: plot failed:", repr(e))

    # random plots with overall avg curve
    if args.plot_random:
        try:
            import matplotlib.pyplot as plt

            rng = np.random.default_rng(args.plot_seed)
            n_windows = min(int(args.plot_n_windows), len(ds_test))
            n_stations = min(int(args.plot_n_stations), len(data.station_ids))
            win_ids = rng.choice(len(ds_test), size=n_windows, replace=False).tolist()
            station_ids = rng.choice(len(data.station_ids), size=n_stations, replace=False).tolist()

            plot_dir = Path(args.plot_dir) if args.plot_dir else out_path.with_name("random_plots")
            plot_dir.mkdir(parents=True, exist_ok=True)

            target_i = AQ_VARS.index(plot_target)

            # keep x_mean/x_std on device for reuse
            x_mean2 = torch.tensor(x_scaler.mean, device=device, dtype=torch.float32).view(1, 1, 1, -1)
            x_std2 = torch.tensor(x_scaler.std, device=device, dtype=torch.float32).view(1, 1, 1, -1)

            for wi in win_ids:
                sample = ds_test[wi]
                x_hist_s = sample["x_hist"].unsqueeze(0).to(device)
                met_all_s = sample["met_all"].unsqueeze(0).to(device)
                y_s = sample["y"].unsqueeze(0).to(device)

                x_hist_den = x_hist_s.float() * x_std2 + x_mean2
                y_den = y_s.float() * x_std2 + x_mean2

                with autocast(autocast_device, enabled=bool(args.amp and device.type == "cuda"), dtype=amp_dtype):
                    out = model(x_hist=x_hist_s, met_all=met_all_s, graph=graph, teacher_forcing=True)
                    pred = out.y_pred
                pred_den = pred.float() * x_std2 + x_mean2

                s = int(ds_test.start_indices[wi])  # type: ignore[attr-defined]
                start_ts = str(data.time[s + args.t_hist - 1]) if (s + args.t_hist - 1) < len(data.time) else f"idx{s}"

                # station-specific plots
                for si in station_ids:
                    sid = data.station_ids[si]
                    hist = x_hist_den[0, :, si, target_i].detach().cpu().numpy()
                    fut_true = y_den[0, :, si, target_i].detach().cpu().numpy()
                    fut_pred = pred_den[0, :, si, target_i].detach().cpu().numpy()
                    t = np.arange(1, len(hist) + len(fut_true) + 1)
                    true_series = np.concatenate([hist, fut_true], axis=0)
                    pred_series = np.concatenate([hist, fut_pred], axis=0)
                    plt.figure(figsize=(10, 4))
                    plt.plot(t, true_series, label="true", linewidth=2)
                    plt.plot(t, pred_series, label="pred", linewidth=2)
                    plt.axvline(len(hist), color="k", linestyle="--", alpha=0.4, label="hist/pred split")
                    plt.title(f"{plot_target} true vs pred @ {sid} (start={start_ts}, seed={args.plot_seed})")
                    plt.xlabel("step (hist+pred)")
                    plt.ylabel(plot_target)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    out_file = plot_dir / f"seed{args.plot_seed}_win{wi}_station{sid}_{plot_target}.png"
                    plt.savefig(out_file, dpi=160)
                    plt.close()

                # overall average plot
                hist_avg = x_hist_den[0, :, :, target_i].mean(dim=1).detach().cpu().numpy()
                fut_true_avg = y_den[0, :, :, target_i].mean(dim=1).detach().cpu().numpy()
                fut_pred_avg = pred_den[0, :, :, target_i].mean(dim=1).detach().cpu().numpy()
                t = np.arange(1, len(hist_avg) + len(fut_true_avg) + 1)
                true_series = np.concatenate([hist_avg, fut_true_avg], axis=0)
                pred_series = np.concatenate([hist_avg, fut_pred_avg], axis=0)
                plt.figure(figsize=(10, 4))
                plt.plot(t, true_series, label="true", linewidth=2)
                plt.plot(t, pred_series, label="pred", linewidth=2)
                plt.axvline(len(hist_avg), color="k", linestyle="--", alpha=0.4, label="hist/pred split")
                plt.title(f"{plot_target} true vs pred @ ALL_STATIONS_AVG (start={start_ts}, seed={args.plot_seed})")
                plt.xlabel("step (hist+pred)")
                plt.ylabel(plot_target)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                out_file = plot_dir / f"seed{args.plot_seed}_win{wi}_stationALL_AVG_{plot_target}.png"
                plt.savefig(out_file, dpi=160)
                plt.close()

            print("random plots saved under:", str(plot_dir))
        except Exception as e:
            print("WARN: random plot failed:", repr(e))


if __name__ == "__main__":
    main()
