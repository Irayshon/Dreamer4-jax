import json
import shutil
import unittest
import uuid
from pathlib import Path

from dreamer.pipeline.plots import HAS_MATPLOTLIB, build_run_dashboard, plot_stage_curves, write_best_checkpoint


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for plotting tests")
class PipelinePlotsTests(unittest.TestCase):
    def test_stage_plots_and_best_checkpoint_generation(self):
        temp_root = (Path.cwd() / ".tmp_tests" / f"dreamer_pipeline_plots_{uuid.uuid4().hex}").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            stage_dir = temp_root / "tokenizer"
            ckpt_dir = stage_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = stage_dir / "metrics.jsonl"
            rows = [
                {"stage": "tokenizer", "step": 0, "loss_total": 1.0, "loss_mse": 0.8, "loss_lpips": 0.2, "psnr": 5.0},
                {"stage": "tokenizer", "step": 10, "loss_total": 0.6, "loss_mse": 0.5, "loss_lpips": 0.1, "psnr": 8.0},
                {"stage": "tokenizer", "step": 20, "loss_total": 0.4, "loss_mse": 0.3, "loss_lpips": 0.08, "psnr": 10.0},
            ]
            with metrics_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")

            curves = plot_stage_curves("tokenizer", stage_dir)
            self.assertIn("overview", curves)
            self.assertIn("loss_trend", curves)
            self.assertTrue(Path(curves["overview"]).exists())
            self.assertTrue(Path(curves["loss_trend"]).exists())

            best = write_best_checkpoint("tokenizer", stage_dir, ckpt_dir)
            self.assertIsNotNone(best)
            self.assertEqual(best["metric_key"], "loss_total")
            self.assertEqual(best["step"], 20)
            self.assertTrue((stage_dir / "best_checkpoint.json").exists())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_run_dashboard_generation(self):
        temp_root = (Path.cwd() / ".tmp_tests" / f"dreamer_pipeline_dashboard_{uuid.uuid4().hex}").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            stage_data = {
                "tokenizer": {"loss_total": [1.0, 0.7, 0.5]},
                "dynamics": {"flow_mse": [0.3, 0.2, 0.1]},
                "bc_rew": {"loss_total": [0.8, 0.6, 0.4]},
                "policy": {"val_loss": [1.2, 1.0, 0.9], "eval/return_mean": [-0.3, -0.1, 0.05]},
            }
            for stage, metrics in stage_data.items():
                stage_dir = temp_root / stage
                stage_dir.mkdir(parents=True, exist_ok=True)
                with (stage_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
                    for step_idx in range(3):
                        row = {"stage": stage, "step": step_idx * 100}
                        for key, values in metrics.items():
                            row[key] = values[step_idx]
                        f.write(json.dumps(row, ensure_ascii=True) + "\n")
                with (stage_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "stage": stage,
                            "metric_key": list(metrics.keys())[0],
                            "mode": "min",
                            "step": 200,
                            "value": 0.1,
                            "checkpoint_path": str(stage_dir / "checkpoints"),
                        },
                        f,
                        ensure_ascii=True,
                    )

            generated = build_run_dashboard(temp_root)
            self.assertIn("dashboard", generated)
            self.assertIn("kpi_table", generated)
            self.assertTrue(Path(generated["dashboard"]).exists())
            self.assertTrue(Path(generated["kpi_table"]).exists())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
