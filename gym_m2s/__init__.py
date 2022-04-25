from gym.envs.registration import register

kwargs = {
    'reward_type': 'sparse'
}

register(
    id='SingleFetchPickAndPlace-v0',
    entry_point='gym_m2s.robotics:StartGoalFixedFetchPickAndPlaceEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)

register(
    id='SingleFetchPickAndPlace-v1',
    entry_point='gym_m2s.robotics:SparseGoalRewardFetchPickAndPlaceEnv',
    kwargs=kwargs,
    max_episode_steps=50
)

register(
    id='SingleFetchPickAndPlace-v2',
    entry_point='gym_m2s.robotics:StartFixedFetchPickAndPlaceEnv',
    kwargs=kwargs,
    max_episode_steps=50
)