import unittest


try:
    import jax
    import jax.numpy as jnp
    from dreamer.envs import (
        get_env_spec,
        make_env_reset_fn,
        make_env_step_fn,
        make_iterator,
        unpack_batch,
    )

    JAX_AVAILABLE = True
except ModuleNotFoundError:
    JAX_AVAILABLE = False


@unittest.skipUnless(JAX_AVAILABLE, "JAX is required for environment smoke tests.")
class EnvironmentContractTests(unittest.TestCase):
    def test_get_env_spec_aliases(self):
        toy = get_env_spec("bouncing_square")
        grasp = get_env_spec("grasping_2p5d")
        self.assertEqual(get_env_spec("grasping").name, grasp.name)
        self.assertEqual(get_env_spec("graspworld").name, grasp.name)
        self.assertEqual(toy.action_dim, 4)
        self.assertEqual(grasp.action_dim, 8)
        self.assertFalse(toy.supports_task_ids)
        self.assertTrue(grasp.supports_task_ids)

    def test_unpack_batch_adds_default_task_ids_for_toy_batches(self):
        frames = jnp.zeros((3, 5, 8, 8, 3), dtype=jnp.float32)
        actions = jnp.zeros((3, 5), dtype=jnp.int32)
        rewards = jnp.zeros((3, 5), dtype=jnp.float32)

        out_frames, out_actions, out_rewards, task_ids = unpack_batch(
            (frames, actions, rewards),
            batch_size=3,
        )

        self.assertEqual(out_frames.shape, frames.shape)
        self.assertEqual(out_actions.shape, actions.shape)
        self.assertEqual(out_rewards.shape, rewards.shape)
        self.assertEqual(task_ids.shape, (3,))
        self.assertTrue(jnp.all(task_ids == 0))

    def test_bouncing_square_iterator_contract(self):
        iterator = make_iterator(
            "bouncing_square",
            batch_size=2,
            time_steps=6,
            height=32,
            width=32,
            channels=3,
        )
        _, batch = iterator(jax.random.PRNGKey(0))
        frames, actions, rewards, task_ids = unpack_batch(batch, batch_size=2)

        self.assertEqual(frames.shape, (2, 6, 32, 32, 3))
        self.assertEqual(actions.shape, (2, 6))
        self.assertEqual(rewards.shape, (2, 6))
        self.assertEqual(task_ids.shape, (2,))

    def test_grasping_iterator_contract(self):
        iterator = make_iterator(
            "grasping_2p5d",
            batch_size=2,
            time_steps=6,
            height=32,
            width=32,
            channels=3,
        )
        _, batch = iterator(jax.random.PRNGKey(0))
        frames, actions, rewards, task_ids = unpack_batch(batch, batch_size=2)

        self.assertEqual(frames.shape, (2, 6, 32, 32, 3))
        self.assertEqual(actions.shape, (2, 6))
        self.assertEqual(rewards.shape, (2, 6))
        self.assertEqual(task_ids.shape, (2,))

    def test_env_reset_and_step_contracts(self):
        toy_reset = make_env_reset_fn(
            "bouncing_square",
            batch_size=2,
            height=32,
            width=32,
            channels=3,
        )
        toy_step = make_env_step_fn("bouncing_square", height=32, width=32, channels=3)
        toy_state, toy_obs0, toy_a0, toy_r0 = toy_reset(jax.random.PRNGKey(1))
        toy_next_state, toy_obs1, toy_reward, toy_done = toy_step(
            toy_state,
            jnp.zeros((2,), dtype=jnp.int32),
        )
        del toy_next_state

        self.assertEqual(toy_obs0.shape, (2, 32, 32, 3))
        self.assertEqual(toy_obs1.shape, (2, 32, 32, 3))
        self.assertEqual(toy_a0.shape, (2,))
        self.assertEqual(toy_r0.shape, (2,))
        self.assertEqual(toy_reward.shape, (2,))
        self.assertEqual(toy_done.shape, (2,))

        grasp_reset = make_env_reset_fn(
            "grasping_2p5d",
            batch_size=2,
            height=32,
            width=32,
            channels=3,
        )
        grasp_step = make_env_step_fn("grasping_2p5d", height=32, width=32, channels=3)
        grasp_state, grasp_obs0, grasp_a0, grasp_r0 = grasp_reset(jax.random.PRNGKey(2))
        grasp_next_state, grasp_obs1, grasp_reward, grasp_done = grasp_step(
            grasp_state,
            jnp.zeros((2,), dtype=jnp.int32),
        )
        del grasp_next_state

        self.assertEqual(grasp_obs0.shape, (2, 32, 32, 3))
        self.assertEqual(grasp_obs1.shape, (2, 32, 32, 3))
        self.assertEqual(grasp_a0.shape, (2,))
        self.assertEqual(grasp_r0.shape, (2,))
        self.assertEqual(grasp_reward.shape, (2,))
        self.assertEqual(grasp_done.shape, (2,))


if __name__ == "__main__":
    unittest.main()
