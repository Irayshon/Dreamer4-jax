import shutil
import unittest
import uuid
from pathlib import Path

from dreamer.pipeline.runner import _resolve_legacy_checkpoint, run_pipeline


class PipelineRunnerTests(unittest.TestCase):
    def test_legacy_checkpoint_resolution(self):
        temp_root = (Path.cwd() / ".tmp_tests" / f"dreamer_pipeline_ckpt_{uuid.uuid4().hex}").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            legacy = temp_root / "logs" / "old_run" / "checkpoints"
            legacy.mkdir(parents=True, exist_ok=True)
            cwd = Path.cwd()
            try:
                # helper checks relative logs/<name>/checkpoints from CWD
                # so run this check inside temp root.
                import os

                os.chdir(temp_root)
                resolved = _resolve_legacy_checkpoint("old_run")
                self.assertEqual(Path(resolved), legacy.resolve())
            finally:
                os.chdir(cwd)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_stage_only_report_writes_manifest_and_summary(self):
        temp_root = (Path.cwd() / ".tmp_tests" / f"dreamer_pipeline_test_{uuid.uuid4().hex}").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            cfg_path = temp_root / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "experiment_name: pipeline_test",
                        f"output_root: {temp_root.as_posix()}",
                        "env_name: grasping_2p5d",
                        "tracking:",
                        "  use_wandb: false",
                        "stages:",
                        "  tokenizer:",
                        "    B: 1",
                        "    T: 2",
                        "    max_steps: 1",
                        "  dynamics:",
                        "    B: 1",
                        "    T: 2",
                        "    max_steps: 1",
                        "  bc_rew:",
                        "    B: 1",
                        "    T: 2",
                        "    max_steps: 1",
                        "  policy:",
                        "    B: 1",
                        "    T: 2",
                        "    max_steps: 1",
                        "  eval:",
                        "  report:",
                    ]
                ),
                encoding="utf-8",
            )
            run_dir = run_pipeline(config_path=cfg_path, command="stage-only", stage_only="report")
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
