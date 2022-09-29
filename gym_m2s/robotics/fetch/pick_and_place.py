from gym.envs.robotics import fetch_env
# from gym_robotics.envs import fetch_env
from gym import utils
from gym.envs.robotics.fetch_env import goal_distance
# from gym_robotics.envs.fetch_env import goal_distance
import os
import numpy as np


MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


class GoalFixedPnPEnv(fetch_env.FetchEnv, utils.EzPickle):
    """Fixed goal position."""
    def __init__(self, reward_type='sparse', initial_goal_seed=None):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.fixed_object_qpos = initial_qpos['object0:joint']
        self.fixed_goal = None
        self.np_goal_random, _ = utils.seeding.np_random(initial_goal_seed)
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type
        )
        utils.EzPickle.__init__(self)

    def _sample_goal(self):
        return self.fixed_goal.copy()

    def _env_setup(self, initial_qpos):
        super()._env_setup(initial_qpos)
        self.fixed_goal = self._initialize_goal()

    def _initialize_goal(self):
        u_random = self.np_goal_random.uniform(
            -self.target_range,
            self.target_range, size=3
        )
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + u_random
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_goal_random.uniform() < 0.5:
                goal[2] += self.np_goal_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + u_random
        return goal.copy()


class StartGoalFixedPnPEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', initial_goal_seed=None):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.fixed_object_qpos = initial_qpos['object0:joint']
        self.fixed_goal = None
        self.np_goal_random, _ = utils.seeding.np_random(initial_goal_seed)
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False,
            n_substeps=20, gripper_extra_height=0.2, target_in_the_air=True,
            target_offset=0.0, obj_range=0.15, target_range=0.15,
            distance_threshold=0.05, initial_qpos=initial_qpos,
            reward_type=reward_type
        )
        utils.EzPickle.__init__(self)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        if self.has_object:
            self.sim.data.set_joint_qpos('object0:joint',
                                         self.fixed_object_qpos)
        self.sim.forward()
        return True

    def _sample_goal(self):
        return self.fixed_goal.copy()

    def _env_setup(self, initial_qpos):
        super()._env_setup(initial_qpos)
        self.fixed_goal = self._initialize_goal()
        self.fixed_object_qpos = self._initialize_object_xpos()

    def _initialize_goal(self):
        u_random = self.np_goal_random.uniform(
            -self.target_range,
            self.target_range, size=3
        )
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + u_random
            goal += self.target_offset
            goal[2] = self.height_offset

            if self.target_in_the_air and self.np_goal_random.uniform() < 0.5:
                goal[2] += self.np_goal_random.uniform(0, 0.45)
            
            goal = np.array([1.20391183, 0.83966603, 0.42469975])
        else:
            goal = self.initial_gripper_xpos[:3] + u_random
        return goal.copy()

    def _initialize_object_xpos(self):
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            u_random = self.np_goal_random.uniform(
                -self.obj_range, self.obj_range, size=2
            )
            object_xpos = self.initial_gripper_xpos[:2] + u_random
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,) and object_xpos.shape == (2,)
        object_qpos[:2] = object_xpos
        # self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        return object_qpos


class StartGoalFixedGoalRewardPnPEnv(StartGoalFixedPnPEnv):
    """start and goal position is fixed, and generate sparse goal positive reward."""
    def __init__(self, reward_type='sparse', initial_goal_seed=None):
        super().__init__(reward_type, initial_goal_seed)
        self.init_obs = None

    def compute_reward(self, achieved_goal, goal, info):
        """Compute goal reward"""
        d = goal_distance(achieved_goal, goal)
        return (d <= self.distance_threshold).astype(np.float32)

    def reset(self):
        """generate observation."""
        obs = super().reset()

        if self.init_obs is None:
            # スタートを生成していなければ、生成して登録。
            self.init_obs = obs

        return self.init_obs


class GoalFixedGoalFixedPnPEnv(GoalFixedPnPEnv):
    def __init__(self, reward_type='sparse', initial_goal_seed=None):
        super().__init__(reward_type, initial_goal_seed)
        self.init_obs = None

    def compute_reward(self, achieved_goal, goal, info):
        """Compute goal reward"""
        d = goal_distance(achieved_goal, goal)
        return (d <= self.distance_threshold).astype(np.float32)


class StartGoalFixedNoRewardPnPEnv(StartGoalFixedPnPEnv):
    """start and goal position is fixed, and generate no rewards."""
    def __init__(self, reward_type='sparse', initial_goal_seed=None):
        super().__init__(reward_type, initial_goal_seed)
        self.init_obs = None

    def compute_reward(self, achieved_goal, goal, info):
        """Compute goal reward"""
        d = goal_distance(achieved_goal, goal)
        return (d <= self.distance_threshold).astype(np.float32)

    def reset(self):
        """generate observation."""
        obs = super().reset()

        if self.init_obs is None:
            # スタートを生成していなければ、生成して登録。
            self.init_obs = obs

        return self.init_obs