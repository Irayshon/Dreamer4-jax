import unittest
from pathlib import Path

from dreamer.pipeline.config import load_config


class PipelineConfigTests(unittest.TestCase):
    def test_quick_profile_loads(self):
        cfg = load_config(Path("configs/profiles/quick_test.yaml"))
        self.assertEqual(cfg["env_name"], "grasping_2p5d")
        self.assertIn("stages", cfg)
        self.assertIn("tokenizer", cfg["stages"])
        self.assertGreater(cfg["stages"]["tokenizer"]["max_steps"], 0)

    def test_production_profile_loads(self):
        cfg = load_config(Path("configs/profiles/production.yaml"))
        self.assertEqual(cfg["env_name"], "grasping_2p5d")
        self.assertGreater(cfg["stages"]["policy"]["max_steps"], cfg["stages"]["tokenizer"]["max_steps"])

    def test_production_policy_recover_profile_loads(self):
        cfg = load_config(Path("configs/profiles/production_policy_recover.yaml"))
        self.assertEqual(cfg["env_name"], "grasping_2p5d")
        policy = cfg["stages"]["policy"]
        self.assertTrue(policy["save_eval_media"])
        self.assertIn("bc_anchor_weight", policy)
        self.assertIn("eval_greedy", policy)


if __name__ == "__main__":
    unittest.main()
