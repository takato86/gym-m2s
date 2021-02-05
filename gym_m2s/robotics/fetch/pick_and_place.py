from gym.envs.robotics import fetch_env
from gym import utils
import os
import numpy as np


MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


class SingleFetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.fixed_object_qpos = initial_qpos['object0:joint']
        self.fixed_goal = None
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False,
            n_substeps=20, gripper_extra_height=0.2, target_in_the_air=True,
            target_offset=0.0, obj_range=0.15, target_range=0.15,
            distance_threshold=0.05, initial_qpos=initial_qpos,
            reward_type=reward_type)
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

    def _initialize_goal(self):
        u_random = self.np_random.uniform(
            -self.target_range,
            self.target_range, size=3
        )
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + u_random
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + u_random
        return goal.copy()

    def _initialize_object_xpos(self):
        object_xpos = self.initial_gripper_xpos[:2]
        u_random = self.np_random.uniform(
            -self.obj_range, self.obj_range, size=2
        )
        l2_dist = np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2])
        while l2_dist < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + u_random
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_qpos
        # self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        return object_qpos
