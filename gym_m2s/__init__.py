from gym.envs.registration import register

kwargs = {
    'reward_type': 'sparse'
}

register(
    id='SingleFetchPickAndPlace-v0',
    entry_point='gym_m2s.robotics:StartGoalFixedPnPEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)

register(
    id='SingleFetchPickAndPlace-v1',
    entry_point='gym_m2s.robotics:StartGoalFixedGoalRewardPnPEnv',
    kwargs=kwargs,
    max_episode_steps=50
)

register(
    id='SingleFetchPickAndPlace-v2',
    entry_point='gym_m2s.robotics:GoalFixedPnPEnv',
    kwargs=kwargs,
    max_episode_steps=50
)

register(
    id='SingleFetchPickAndPlace-v3',
    entry_point='gym_m2s.robotics:GoalFixedGoalFixedPnPEnv',
    kwargs=kwargs,
    max_episode_steps=50
)

register(
    id='FetchPickAndPlaceNoReward-v0',
    entry_point='gym_m2s.robotics:StartGoalFixedNoRewardPnPEnv',
    kwargs=kwargs,
    max_episode_steps=50
)