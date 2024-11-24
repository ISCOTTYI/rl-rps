import numpy as np
from rps_game.envs.grid_world import RPSEnv
import time


if __name__ == "__main__":
    env = RPSEnv(render_mode="human")
    observation, info = env.reset(seed=42)

    for _ in range(100):
        mask = env.compute_action_mask()
        valid_actions = np.argwhere(mask == True).flatten()
        action = valid_actions[0]
        # print(env._action_to_move(action))
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(1)
    
    # for _ in range(1000):
    #     # this is where you would insert your policy
    #     action = env.action_space.sample()

    #     # step (transition) through the environment with the action
    #     # receiving the next observation, reward and if the episode has terminated or truncated
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     # If the episode has ended then we can reset to start a new episode
    #     if terminated or truncated:
    #         observation, info = env.reset()

    env.close()
