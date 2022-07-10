import gym
import gym_m2s
import unittest

gym_m2s


def basic_flow(env):
    env.seed(123)
    obs = env.reset()
    a = env.action_space.sample()
    obs, _, _, _ = env.step(a)
    achieved_goal = obs["achieved_goal"]
    desired_goal = obs["desired_goal"]
    return obs, achieved_goal, desired_goal, a


class PnPEnvTest(unittest.TestCase):
    def test_initialize_v0(self):
        env = gym.make("SingleFetchPickAndPlace-v0")
        obs, a_goal, d_goal, a = basic_flow(env)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(a_goal), 3)
        self.assertEqual(len(d_goal), 3)
        self.assertEqual(len(obs["observation"]), 25)

    def test_initialize_v1(self):
        env = gym.make("SingleFetchPickAndPlace-v1")
        obs, a_goal, d_goal, a = basic_flow(env)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(a_goal), 3)
        self.assertEqual(len(d_goal), 3)
        self.assertEqual(len(obs["observation"]), 25)

    def test_initialize_v2(self):
        env = gym.make("SingleFetchPickAndPlace-v2")
        obs, a_goal, d_goal, a = basic_flow(env)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(a_goal), 3)
        self.assertEqual(len(d_goal), 3)
        self.assertEqual(len(obs["observation"]), 25)

    def test_initialize_v3(self):
        env = gym.make("SingleFetchPickAndPlace-v3")
        obs, a_goal, d_goal, a = basic_flow(env)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(a_goal), 3)
        self.assertEqual(len(d_goal), 3)
        self.assertEqual(len(obs["observation"]), 25)

    def test_initialize_no_reward(self):
        env = gym.make("FetchPickAndPlaceNoReward-v0")
        obs, a_goal, d_goal, a = basic_flow(env)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(a_goal), 3)
        self.assertEqual(len(d_goal), 3)
        self.assertEqual(len(obs["observation"]), 25)


if __name__ == "__main__":
    unittest.main()
