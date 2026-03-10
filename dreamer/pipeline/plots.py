from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


TRAINING_STAGES = ("tokenizer", "dynamics", "bc_rew", "policy")

PRIMARY_METRIC_BY_STAGE: dict[str, tuple[str, str]] = {
    "tokenizer": ("loss_total", "min"),
    "dynamics": ("flow_mse", "min"),
    "bc_rew": ("loss_total", "min"),
    "policy": ("eval/return_mean", "max"),
}

OVERVIEW_KEYS_BY_STAGE: dict[str, list[str]] = {
    "tokenizer": ["loss_total", "loss_mse", "loss_lpips", "psnr"],
    "dynamics": ["flow_mse", "bootstrap_mse", "loss_total"],
    "bc_rew": ["loss_total", "flow_mse", "w_pi_ce", "w_rw_ce"],
    "policy": ["val_loss", "pi_loss", "pi_kl_loss", "eval/return_mean"],
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def _to_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _get_series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for idx, row in enumerate(rows):
        y = _to_float(row.get(key))
        if y is None:
            continue
        x_raw = row.get("step", idx)
        x = int(x_raw) if isinstance(x_raw, (int, float)) else idx
        xs.append(x)
        ys.append(y)
    return xs, ys


def _ema(values: list[float], alpha: float = 0.15) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def plot_stage_curves(stage: str, stage_dir: Path) -> dict[str, str]:
    if not HAS_MATPLOTLIB:
        return {}
    metrics_path = stage_dir / "metrics.jsonl"
    rows = _read_jsonl(metrics_path)
    if not rows:
        return {}

    curves_dir = stage_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    keys = [key for key in OVERVIEW_KEYS_BY_STAGE.get(stage, []) if _get_series(rows, key)[0]]
    if keys:
        n_cols = 2
        n_rows = (len(keys) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.5 * n_rows))
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, key in zip(axes_list, keys):
            xs, ys = _get_series(rows, key)
            ax.plot(xs, ys, color="#2E86AB", linewidth=1.4, label=key)
            ax.set_title(key)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
        for ax in axes_list[len(keys) :]:
            ax.axis("off")
        fig.suptitle(f"{stage} overview", fontsize=12)
        fig.tight_layout()
        overview_path = curves_dir / "overview.png"
        fig.savefig(overview_path, dpi=150)
        plt.close(fig)
        generated["overview"] = str(overview_path)

    primary = PRIMARY_METRIC_BY_STAGE.get(stage, ("loss_total", "min"))[0]
    xs, ys = _get_series(rows, primary)
    if xs and ys:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, ys, color="#7B2CBF", linewidth=1.0, alpha=0.4, label=f"{primary} raw")
        ax.plot(xs, _ema(ys), color="#1B9E77", linewidth=2.0, label=f"{primary} EMA")
        ax.set_title(f"{stage} loss trend")
        ax.set_xlabel("step")
        ax.set_ylabel(primary)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        loss_path = curves_dir / "loss_trend.png"
        fig.savefig(loss_path, dpi=150)
        plt.close(fig)
        generated["loss_trend"] = str(loss_path)

    return generated


def _best_from_rows(rows: list[dict[str, Any]], metric_key: str, mode: str) -> tuple[int | None, float | None]:
    best_step: int | None = None
    best_value: float | None = None
    for idx, row in enumerate(rows):
        value = _to_float(row.get(metric_key))
        if value is None:
            continue
        step_raw = row.get("step", idx)
        step = int(step_raw) if isinstance(step_raw, (int, float)) else idx
        if best_value is None:
            best_value = value
            best_step = step
            continue
        if mode == "min" and value < best_value:
            best_value = value
            best_step = step
        if mode == "max" and value > best_value:
            best_value = value
            best_step = step
    return best_step, best_value


def write_best_checkpoint(stage: str, stage_dir: Path, checkpoint_dir: Path | None) -> dict[str, Any] | None:
    if stage not in TRAINING_STAGES:
        return None
    rows = _read_jsonl(stage_dir / "metrics.jsonl")
    if not rows:
        return None

    metric_key, mode = PRIMARY_METRIC_BY_STAGE.get(stage, ("loss_total", "min"))
    if stage == "policy":
        eval_step, eval_value = _best_from_rows(rows, "eval/return_mean", "max")
        if eval_step is not None:
            metric_key, mode = "eval/return_mean", "max"
            best_step, best_value = eval_step, eval_value
        else:
            metric_key, mode = "val_loss", "min"
            best_step, best_value = _best_from_rows(rows, metric_key, mode)
    else:
        best_step, best_value = _best_from_rows(rows, metric_key, mode)

    if best_step is None or best_value is None:
        return None

    payload = {
        "stage": stage,
        "metric_key": metric_key,
        "mode": mode,
        "step": int(best_step),
        "value": float(best_value),
        "checkpoint_path": str(checkpoint_dir) if checkpoint_dir is not None else None,
    }
    out_path = stage_dir / "best_checkpoint.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return payload


def build_run_dashboard(run_dir: Path) -> dict[str, str]:
    if not HAS_MATPLOTLIB:
        return {}
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes_list = axes.flatten()
    any_series = False
    for idx, stage in enumerate(TRAINING_STAGES):
        ax = axes_list[idx]
        rows = _read_jsonl(run_dir / stage / "metrics.jsonl")
        metric_key = PRIMARY_METRIC_BY_STAGE[stage][0]
        if stage == "policy":
            xs, ys = _get_series(rows, "eval/return_mean")
            if not xs:
                xs, ys = _get_series(rows, "val_loss")
                metric_key = "val_loss"
        else:
            xs, ys = _get_series(rows, metric_key)
        if xs:
            any_series = True
            ax.plot(xs, ys, color="#2F4858", linewidth=1.0, alpha=0.35, label="raw")
            ax.plot(xs, _ema(ys), color="#F26419", linewidth=2.0, label="EMA")
            ax.set_title(f"{stage}: {metric_key}")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
        else:
            ax.set_title(f"{stage}: no metrics")
            ax.axis("off")
    fig.tight_layout()
    dashboard_path = metrics_dir / "dashboard.png"
    fig.savefig(dashboard_path, dpi=150)
    plt.close(fig)
    if any_series:
        generated["dashboard"] = str(dashboard_path)

    rows = []
    for stage in TRAINING_STAGES:
        best_path = run_dir / stage / "best_checkpoint.json"
        if not best_path.exists():
            continue
        with best_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        rows.append(
            [
                stage,
                str(loaded.get("metric_key", "")),
                str(loaded.get("step", "")),
                f"{float(loaded.get('value', float('nan'))):.6g}",
            ]
        )

    if rows:
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.9 + 0.5 * len(rows))))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=["Stage", "Metric", "Best Step", "Best Value"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.3)
        fig.tight_layout()
        kpi_path = metrics_dir / "kpi_table.png"
        fig.savefig(kpi_path, dpi=160)
        plt.close(fig)
        generated["kpi_table"] = str(kpi_path)

    return generated
