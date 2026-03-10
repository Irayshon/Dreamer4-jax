from __future__ import annotations

import dataclasses
import math
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dreamer.pipeline.config import STAGE_ORDER, compute_paths, dump_yaml, load_config
from dreamer.pipeline.io import StageRecord, append_jsonl, ensure_dirs, now_iso, read_json, write_json
from dreamer.pipeline.plots import build_run_dashboard, plot_stage_curves, write_best_checkpoint


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _to_dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    field_names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in values.items() if k in field_names}


def _resolve_legacy_checkpoint(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    maybe = Path("logs") / path / "checkpoints"
    if maybe.exists():
        return str(maybe.resolve())
    return str(p)


def _load_last_metrics(policy_metrics_jsonl: Path) -> dict[str, float]:
    defaults = {
        "eval/return_mean": float("nan"),
        "eval/return_std": float("nan"),
        "eval/grasp_success_rate": float("nan"),
        "eval/place_success_rate": float("nan"),
        "eval/attach_steps_mean": float("nan"),
        "eval/final_goal_distance_mean": float("nan"),
        "eval/close_count_mean": float("nan"),
        "eval/lower_count_mean": float("nan"),
        "eval/lift_count_mean": float("nan"),
        "eval/near_object_steps_mean": float("nan"),
        "eval/grasp_attempt_count_mean": float("nan"),
        "eval/attached_ratio_mean": float("nan"),
        "eval/goal_chase_while_unattached_steps_mean": float("nan"),
    }
    if not policy_metrics_jsonl.exists():
        return defaults

    last_line = None
    with policy_metrics_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line
    if last_line is None:
        return defaults

    out = dict(defaults)
    lines = policy_metrics_jsonl.read_text(encoding="utf-8").splitlines()
    for raw in reversed(lines):
        if not raw.strip():
            continue
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            continue
        found = False
        for key in out:
            value = loaded.get(key)
            if value is not None:
                out[key] = float(value)
                found = True
        if found:
            return out
    return out


def _load_best_checkpoint(run_dir: Path, stage: str) -> dict[str, Any] | None:
    best_path = run_dir / stage / "best_checkpoint.json"
    if not best_path.exists():
        return None
    loaded = read_json(best_path)
    return loaded if isinstance(loaded, dict) else None


def _write_summary(run_dir: Path, cfg: dict[str, Any], manifest: dict[str, Any]) -> None:
    env_name = cfg["env_name"]
    metrics = _load_last_metrics(run_dir / "policy" / "metrics.jsonl")
    lines = [
        f"# Experiment Summary: {cfg['experiment_name']}",
        "",
        f"- Environment: `{env_name}`",
        f"- Run directory: `{run_dir}`",
        "",
        "## Stage Status",
        "",
    ]

    for stage in STAGE_ORDER:
        record = manifest.get("stages", {}).get(stage)
        status = "pending" if record is None else str(record.get("status", "unknown"))
        lines.append(f"- {stage}: {status}")

    lines.extend(
        [
            "",
            "## Final Eval Metrics",
            "",
            f"- return_mean: {metrics['eval/return_mean']:.6g}" if not math.isnan(metrics["eval/return_mean"]) else "- return_mean: NaN",
            f"- return_std: {metrics['eval/return_std']:.6g}" if not math.isnan(metrics["eval/return_std"]) else "- return_std: NaN",
            f"- grasp_success_rate: {metrics['eval/grasp_success_rate']:.6g}" if not math.isnan(metrics["eval/grasp_success_rate"]) else "- grasp_success_rate: NaN",
            f"- place_success_rate: {metrics['eval/place_success_rate']:.6g}" if not math.isnan(metrics["eval/place_success_rate"]) else "- place_success_rate: NaN",
            f"- attach_steps_mean: {metrics['eval/attach_steps_mean']:.6g}" if not math.isnan(metrics["eval/attach_steps_mean"]) else "- attach_steps_mean: NaN",
            f"- final_goal_distance_mean: {metrics['eval/final_goal_distance_mean']:.6g}" if not math.isnan(metrics["eval/final_goal_distance_mean"]) else "- final_goal_distance_mean: NaN",
            "",
            "## Grasp Behavior Diagnostics",
            "",
            f"- close_count_mean: {metrics['eval/close_count_mean']:.6g}" if not math.isnan(metrics["eval/close_count_mean"]) else "- close_count_mean: NaN",
            f"- lower_count_mean: {metrics['eval/lower_count_mean']:.6g}" if not math.isnan(metrics["eval/lower_count_mean"]) else "- lower_count_mean: NaN",
            f"- lift_count_mean: {metrics['eval/lift_count_mean']:.6g}" if not math.isnan(metrics["eval/lift_count_mean"]) else "- lift_count_mean: NaN",
            f"- near_object_steps_mean: {metrics['eval/near_object_steps_mean']:.6g}" if not math.isnan(metrics["eval/near_object_steps_mean"]) else "- near_object_steps_mean: NaN",
            f"- grasp_attempt_count_mean: {metrics['eval/grasp_attempt_count_mean']:.6g}" if not math.isnan(metrics["eval/grasp_attempt_count_mean"]) else "- grasp_attempt_count_mean: NaN",
            f"- attached_ratio_mean: {metrics['eval/attached_ratio_mean']:.6g}" if not math.isnan(metrics["eval/attached_ratio_mean"]) else "- attached_ratio_mean: NaN",
            f"- goal_chase_while_unattached_steps_mean: {metrics['eval/goal_chase_while_unattached_steps_mean']:.6g}" if not math.isnan(metrics["eval/goal_chase_while_unattached_steps_mean"]) else "- goal_chase_while_unattached_steps_mean: NaN",
            "",
            "## Best Checkpoints",
            "",
        ]
    )

    for stage in ("tokenizer", "dynamics", "bc_rew", "policy"):
        best = _load_best_checkpoint(run_dir, stage)
        if best is None:
            lines.append(f"- {stage}: N/A")
            continue
        metric_key = str(best.get("metric_key", "metric"))
        step = best.get("step", "N/A")
        value = best.get("value", "N/A")
        checkpoint_path = best.get("checkpoint_path", "")
        lines.append(
            f"- {stage}: {metric_key}={value} at step={step} ({checkpoint_path})"
        )
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _build_stage_common(stage_cfg: dict[str, Any], run_dir: Path, stage_name: str, env_name: str, use_wandb: bool) -> dict[str, Any]:
    return {
        **stage_cfg,
        "run_name": stage_name,
        "log_dir": str(run_dir),
        "env_name": env_name,
        "use_wandb": use_wandb,
    }


def _run_stage(stage: str, cfg: dict[str, Any], run_dir: Path, manifest: dict[str, Any]) -> None:
    env_name = cfg["env_name"]
    use_wandb = bool(cfg.get("tracking", {}).get("use_wandb", False))
    stages = cfg["stages"]

    stage_record = StageRecord(name=stage, status="running", started_at=now_iso())
    manifest["stages"][stage] = dataclasses.asdict(stage_record)
    write_json(run_dir / "manifest.json", manifest)

    if stage == "tokenizer":
        from scripts.train_tokenizer import TokenizerConfig, run as run_tokenizer

        stage_cfg = _build_stage_common(stages["tokenizer"], run_dir, "tokenizer", env_name, use_wandb)
        tokenizer_cfg = TokenizerConfig(**_to_dataclass_kwargs(TokenizerConfig, stage_cfg))
        run_tokenizer(tokenizer_cfg)
        checkpoint_dir = run_dir / "tokenizer" / "checkpoints"
        details = {"max_steps": tokenizer_cfg.max_steps}

    elif stage == "dynamics":
        from scripts.train_dynamics import RealismConfig, run as run_dynamics

        tok_ckpt = stages["dynamics"].get("tokenizer_ckpt") or str(run_dir / "tokenizer" / "checkpoints")
        stage_cfg = _build_stage_common(stages["dynamics"], run_dir, "dynamics", env_name, use_wandb)
        stage_cfg["tokenizer_ckpt"] = _resolve_legacy_checkpoint(tok_ckpt)
        dyn_cfg = RealismConfig(**_to_dataclass_kwargs(RealismConfig, stage_cfg))
        run_dynamics(dyn_cfg)
        checkpoint_dir = run_dir / "dynamics" / "checkpoints"
        details = {"max_steps": dyn_cfg.max_steps}

    elif stage == "bc_rew":
        from scripts.train_bc_rew_heads import RealismConfig, run as run_bc_rew

        tok_ckpt = stages["bc_rew"].get("tokenizer_ckpt") or str(run_dir / "tokenizer" / "checkpoints")
        dyn_ckpt = stages["bc_rew"].get("pretrained_dyn_ckpt") or str(run_dir / "dynamics" / "checkpoints")
        stage_cfg = _build_stage_common(stages["bc_rew"], run_dir, "bc_rew", env_name, use_wandb)
        stage_cfg["tokenizer_ckpt"] = _resolve_legacy_checkpoint(tok_ckpt)
        stage_cfg["pretrained_dyn_ckpt"] = _resolve_legacy_checkpoint(dyn_ckpt)
        bc_cfg = RealismConfig(**_to_dataclass_kwargs(RealismConfig, stage_cfg))
        run_bc_rew(bc_cfg)
        checkpoint_dir = run_dir / "bc_rew" / "checkpoints"
        details = {"max_steps": bc_cfg.max_steps}

    elif stage == "policy":
        from scripts.train_policy import RLConfig, run as run_policy

        bc_ckpt = stages["policy"].get("bc_rew_ckpt") or str(run_dir / "bc_rew" / "checkpoints")
        stage_cfg = _build_stage_common(stages["policy"], run_dir, "policy", env_name, use_wandb)
        stage_cfg["bc_rew_ckpt"] = _resolve_legacy_checkpoint(bc_ckpt)
        pol_cfg = RLConfig(**_to_dataclass_kwargs(RLConfig, stage_cfg))
        run_policy(pol_cfg)
        checkpoint_dir = run_dir / "policy" / "checkpoints"
        details = {"max_steps": pol_cfg.max_steps}

    elif stage == "eval":
        policy_metrics = run_dir / "policy" / "metrics.jsonl"
        values = _load_last_metrics(policy_metrics)
        standardized = {
            "stage": "eval",
            "env_name": env_name,
            "timestamp": now_iso(),
            "return_mean": values["eval/return_mean"],
            "return_std": values["eval/return_std"],
            "grasp_success_rate": values["eval/grasp_success_rate"],
            "place_success_rate": values["eval/place_success_rate"],
            "attach_steps_mean": values["eval/attach_steps_mean"],
            "final_goal_distance_mean": values["eval/final_goal_distance_mean"],
            "close_count_mean": values["eval/close_count_mean"],
            "lower_count_mean": values["eval/lower_count_mean"],
            "lift_count_mean": values["eval/lift_count_mean"],
            "near_object_steps_mean": values["eval/near_object_steps_mean"],
            "grasp_attempt_count_mean": values["eval/grasp_attempt_count_mean"],
            "attached_ratio_mean": values["eval/attached_ratio_mean"],
            "goal_chase_while_unattached_steps_mean": values["eval/goal_chase_while_unattached_steps_mean"],
        }
        append_jsonl(run_dir / "metrics" / "metrics.jsonl", standardized)
        checkpoint_dir = None
        details = standardized

    elif stage == "report":
        checkpoint_dir = None
        details = {"summary_path": str(run_dir / "summary.md")}

    else:
        raise ValueError(f"Unsupported stage: {stage}")

    stage_record.status = "completed"
    stage_record.finished_at = now_iso()
    stage_record.checkpoint_dir = str(checkpoint_dir) if checkpoint_dir is not None else None
    stage_dir = run_dir / stage
    if stage in {"tokenizer", "dynamics", "bc_rew", "policy"}:
        curves = plot_stage_curves(stage, stage_dir)
        if curves:
            details["curves"] = curves
        best = write_best_checkpoint(stage, stage_dir, checkpoint_dir)
        if best is not None:
            details["best_checkpoint"] = best
    stage_record.details = details
    manifest["stages"][stage] = dataclasses.asdict(stage_record)
    write_json(run_dir / "manifest.json", manifest)
    _write_summary(run_dir, cfg, manifest)
    dashboard = build_run_dashboard(run_dir)
    if dashboard:
        manifest["dashboard"] = dashboard
        write_json(run_dir / "manifest.json", manifest)


def _create_run_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["output_root"]).resolve()
    run_dir = root / cfg["experiment_name"] / _timestamp()
    paths = compute_paths(run_dir)
    ensure_dirs(paths.run_dir, paths.checkpoints_dir, paths.metrics_dir, paths.media_dir)
    (run_dir / "latest_run.txt").write_text(str(run_dir.resolve()), encoding="utf-8")
    return run_dir


def run_pipeline(
    *,
    config_path: str | Path | None = None,
    command: str = "run",
    stage_only: str | None = None,
    run_dir: str | Path | None = None,
    output_root_override: str | None = None,
) -> Path:
    if command not in {"run", "resume", "stage-only"}:
        raise ValueError(f"Unsupported command={command!r}")

    if command in {"run", "stage-only"}:
        if config_path is None:
            raise ValueError("config_path is required for run/stage-only")
        cfg = load_config(config_path)
        if output_root_override:
            cfg = dict(cfg)
            cfg["output_root"] = output_root_override
        current_run_dir = _create_run_dir(cfg)
        (current_run_dir / "config_resolved.yaml").write_text(dump_yaml(cfg), encoding="utf-8")
        manifest = {
            "experiment_name": cfg["experiment_name"],
            "env_name": cfg["env_name"],
            "created_at": now_iso(),
            "config_path": str(Path(config_path).resolve()),
            "stages": {},
        }
        write_json(current_run_dir / "manifest.json", manifest)
    else:
        if run_dir is None:
            raise ValueError("run_dir is required for resume")
        current_run_dir = Path(run_dir).resolve()
        (current_run_dir / "latest_run.txt").write_text(str(current_run_dir.resolve()), encoding="utf-8")
        cfg = load_config(current_run_dir / "config_resolved.yaml")
        manifest = read_json(current_run_dir / "manifest.json")

    selected = [stage_only] if stage_only else STAGE_ORDER
    for stage in selected:
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage}")
    for stage in selected:
        _run_stage(stage, cfg, current_run_dir, manifest)

    return current_run_dir
