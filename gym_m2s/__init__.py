from gym.envs.registration import register

kwargs = {
    'reward_type': 'sparse'
}

register(
    id='SingleFetchPickAndPlace-v0',
    entry_point='gym_m2s.robotics:SingleFetchPickAndPlaceEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)
