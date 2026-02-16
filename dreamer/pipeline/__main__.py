from __future__ import annotations

import argparse
from pathlib import Path

from dreamer.pipeline.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dreamer4 unified pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    run_parser.add_argument("--stage-only", default=None, help="Optional single stage to run")

    stage_parser = sub.add_parser("stage-only", help="Run only one stage in a new run directory")
    stage_parser.add_argument("--config", required=True, help="Path to YAML config")
    stage_parser.add_argument("--stage", required=True, help="Stage name")

    resume_parser = sub.add_parser("resume", help="Resume from existing run directory")
    resume_parser.add_argument("--run-dir", required=True, help="Run directory containing manifest/config")
    resume_parser.add_argument("--stage-only", default=None, help="Optional single stage to run")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_dir = run_pipeline(config_path=Path(args.config), command="run", stage_only=args.stage_only)
    elif args.command == "stage-only":
        run_dir = run_pipeline(config_path=Path(args.config), command="stage-only", stage_only=args.stage)
    else:
        run_dir = run_pipeline(command="resume", run_dir=Path(args.run_dir), stage_only=args.stage_only)
    print(f"[pipeline] Completed. Run dir: {run_dir}")


if __name__ == "__main__":
    main()

