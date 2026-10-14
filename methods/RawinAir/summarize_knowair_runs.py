from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def mean_std(rows: list[dict[str, float]], key: str) -> tuple[float, float]:
    vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def selected_horizons(max_h: int) -> list[int]:
    base = [3, 6, 9, 12, 24]
    hs = [h for h in base if h <= max_h]
    if max_h >= 1 and max_h not in hs:
        hs.append(max_h)
    return hs


def main() -> None:
    run_root = Path(os.environ["RUN_ROOT"])
    horizon_len = int(os.environ.get("T_PRED", "24"))
    runs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run")])
    if not runs:
        raise SystemExit("No run directories found")

    rows: list[dict[str, float]] = []
    for run_dir in runs:
        d = json.loads((run_dir / "eval.json").read_text())
        row: dict[str, float] = {
            "run": run_dir.name,
            "pm25_mae": d.get("PM2.5_mae", np.nan),
            "pm25_rmse": d.get("PM2.5_rmse", np.nan),
            "pm25_smape": d.get("PM2.5_smape", np.nan),
            "pm25_mae_avg": d.get("PM2.5_mae_avg", np.nan),
            "pm25_rmse_avg": d.get("PM2.5_rmse_avg", np.nan),
            "pm25_smape_avg": d.get("PM2.5_smape_avg", np.nan),
            "sequence_mae": d.get("sequence_mae", np.nan),
            "sequence_rmse": d.get("sequence_rmse", np.nan),
        }
        for metric in ["mae_t", "rmse_t", "smape_t"]:
            arr = d.get(f"PM2.5_{metric}", None)
            if arr:
                metric_name = metric.replace("_t", "")
                max_h = min(len(arr), horizon_len)
                for h in [3, 6, 9, 12, 24]:
                    if h <= max_h:
                        row[f"pm25_{metric_name}_{h}"] = arr[h - 1]
                if max_h not in [3, 6, 9, 12, 24]:
                    row[f"pm25_{metric_name}_{max_h}"] = arr[max_h - 1]
                if np.isnan(row.get(f"pm25_{metric_name}_avg", np.nan)):
                    row[f"pm25_{metric_name}_avg"] = float(np.mean(arr[:max_h]))
        rows.append(row)

    summary_lines = [f"Results for {len(rows)} runs", ""]
    for row in rows:
        summary_lines.append(
            f"{row['run']}: PM2.5 mae={row['pm25_mae']:.4f} rmse={row['pm25_rmse']:.4f} smape={row['pm25_smape']:.4f}"
        )
        if not np.isnan(row.get("sequence_mae", np.nan)) and not np.isnan(row.get("sequence_rmse", np.nan)):
            summary_lines.append(
                f"  Full-sequence: mae={row['sequence_mae']:.4f} rmse={row['sequence_rmse']:.4f}"
            )
        for label in ["rmse", "mae", "smape"]:
            parts: list[str] = []
            for h in selected_horizons(horizon_len):
                key = f"pm25_{label}_{h}"
                if key in row:
                    parts.append(f"h{h}={row[key]:.4f}")
            avg_key = f"pm25_{label}_avg"
            if avg_key in row:
                parts.append(f"avg{horizon_len}={row[avg_key]:.4f}")
            if parts:
                summary_lines.append(f"  PM2.5 horizon {label}: " + " ".join(parts))
        summary_lines.append("")

    for key, label in [
        ("pm25_mae", "PM2.5 MAE"),
        ("pm25_rmse", "PM2.5 RMSE"),
        ("pm25_smape", "PM2.5 SMAPE"),
        ("pm25_mae_avg", f"PM2.5 MAE avg{horizon_len}"),
        ("pm25_rmse_avg", f"PM2.5 RMSE avg{horizon_len}"),
        ("pm25_smape_avg", f"PM2.5 SMAPE avg{horizon_len}"),
    ]:
        mu, sd = mean_std(rows, key)
        summary_lines.append(f"{label}: mean={mu:.4f} std={sd:.4f}")

    for key, label in [
        ("sequence_mae", "Full-sequence MAE"),
        ("sequence_rmse", "Full-sequence RMSE"),
    ]:
        mu, sd = mean_std(rows, key)
        summary_lines.append(f"{label}: mean={mu:.4f} std={sd:.4f}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    summary_path = run_root / "summary_3runs.txt"
    summary_path.write_text(summary_text)
    print(f"summary saved: {summary_path}")


if __name__ == "__main__":
    main()
