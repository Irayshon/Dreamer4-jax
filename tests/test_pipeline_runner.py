import shutil
import unittest
import uuid
from pathlib import Path

from dreamer.pipeline.runner import _load_last_metrics, _resolve_legacy_checkpoint, run_pipeline


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
            self.assertTrue((run_dir / "latest_run.txt").exists())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_output_root_override_takes_precedence(self):
        temp_root = (Path.cwd() / ".tmp_tests" / f"dreamer_pipeline_override_{uuid.uuid4().hex}").resolve()
        override_root = (temp_root / "override").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            cfg_path = temp_root / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "experiment_name: pipeline_override_test",
                        f"output_root: {(temp_root / 'from_config').as_posix()}",
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
            run_dir = run_pipeline(
                config_path=cfg_path,
                command="stage-only",
                stage_only="report",
                output_root_override=str(override_root),
            )
            self.assertTrue(str(run_dir).startswith(str(override_root)))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_load_last_metrics_includes_grasp_diagnostics(self):
        temp_root = (Path.cwd() / ".tmp_tests" / f"dreamer_pipeline_metrics_{uuid.uuid4().hex}").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            policy_metrics = temp_root / "policy_metrics.jsonl"
            policy_metrics.write_text(
                "\n".join(
                    [
                        '{"stage":"policy","step":100,"val_loss":1.0}',
                        '{"stage":"policy_eval","step":200,"eval/return_mean":0.1,"eval/return_std":0.2,"eval/grasp_success_rate":0.3,"eval/place_success_rate":0.15,"eval/attach_steps_mean":2.0,"eval/final_goal_distance_mean":5.0,"eval/close_count_mean":4.0,"eval/lower_count_mean":3.0,"eval/lift_count_mean":2.0,"eval/near_object_steps_mean":7.0,"eval/grasp_attempt_count_mean":1.0,"eval/attached_ratio_mean":0.25,"eval/goal_chase_while_unattached_steps_mean":8.0}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = _load_last_metrics(policy_metrics)
            self.assertAlmostEqual(loaded["eval/grasp_success_rate"], 0.3)
            self.assertAlmostEqual(loaded["eval/place_success_rate"], 0.15)
            self.assertAlmostEqual(loaded["eval/close_count_mean"], 4.0)
            self.assertAlmostEqual(loaded["eval/grasp_attempt_count_mean"], 1.0)
            self.assertAlmostEqual(loaded["eval/goal_chase_while_unattached_steps_mean"], 8.0)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
