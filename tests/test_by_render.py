import gym
import gym_m2s
import time

for seed in [10, 20, 30, 40, 50]:
    env = gym.make("SingleFetchPickAndPlace-v0", initial_goal_seed=seed)
    # env.reset()
    print(env.goal)
    # env.render()
    # time.sleep(1)
    env.close()
