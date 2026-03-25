from __future__ import annotations

import argparse
from pathlib import Path

from dreamer.pipeline.runner import run_pipeline, run_visualize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dreamer4 unified pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    run_parser.add_argument("--stage-only", default=None, help="Optional single stage to run")
    run_parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root override (useful for Colab Drive mounts)",
    )

    stage_parser = sub.add_parser("stage-only", help="Run only one stage in a new run directory")
    stage_parser.add_argument("--config", required=True, help="Path to YAML config")
    stage_parser.add_argument("--stage", required=True, help="Stage name")
    stage_parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root override (useful for Colab Drive mounts)",
    )

    resume_parser = sub.add_parser("resume", help="Resume from existing run directory")
    resume_parser.add_argument("--run-dir", required=True, help="Run directory containing manifest/config")
    resume_parser.add_argument("--stage-only", default=None, help="Optional single stage to run")

    visualize_parser = sub.add_parser("visualize", help="Re-generate stage visualization from a run checkpoint")
    visualize_parser.add_argument("--run-dir", required=True, help="Run directory containing resolved config and checkpoints")
    visualize_parser.add_argument("--stage", required=True, choices=["dynamics", "bc_rew", "policy"], help="Stage to visualize")
    visualize_parser.add_argument(
        "--ckpt",
        default="latest",
        help="Checkpoint step (integer) or 'latest' (default)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_dir = run_pipeline(
            config_path=Path(args.config),
            command="run",
            stage_only=args.stage_only,
            output_root_override=args.output_root,
        )
    elif args.command == "stage-only":
        run_dir = run_pipeline(
            config_path=Path(args.config),
            command="stage-only",
            stage_only=args.stage,
            output_root_override=args.output_root,
        )
    elif args.command == "resume":
        run_dir = run_pipeline(command="resume", run_dir=Path(args.run_dir), stage_only=args.stage_only)
        print(f"[pipeline] Completed. Run dir: {run_dir}")
    else:
        summary = run_visualize(
            run_dir=Path(args.run_dir),
            stage=args.stage,
            ckpt=args.ckpt,
        )
        print(f"[pipeline] Visualize completed. stage={args.stage} ckpt={args.ckpt}")
        if isinstance(summary, dict):
            print(f"[pipeline] Visualize summary keys: {sorted(summary.keys())}")
        return
    print(f"[pipeline] Completed. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
