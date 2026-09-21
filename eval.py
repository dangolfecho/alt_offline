import d3rlpy
import argparse
import onnxscript
import gymnasium as gym
import PyFlyt.gym_envs
import pandas as pd
import numpy as np

DEFAULT_NAME = "PPO_CQL"
DEFAULT_NUM = int(2e6)

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]

def main(mname=DEFAULT_NAME):
    iter_num = DEFAULT_NUM
    #save_path = f'backup/plot_cache/{mname}/model_' 
    save_path = f'models/{mname}/model_{iter_num}.d3'
    env = gym.make(envs[0])
    model = d3rlpy.load_learnable(save_path)

    ep_count = 0
    observation = np.random.random((1, 3))
    observation = env.reset()
    rewards = [[]]
    while ep_count < 10:
        obs = observation[0][:, None]
        action = model.predict(obs)[0]
        observation, reward, done, _ = env.step(action)
        rewards[ep_count].append(reward)
        if done:
            observation = env.reset()
            ep_count += 1
            rewards.append([])

    df = pd.DataFrame(reward)
    df.to_csv("rewards.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='eval.py',
            description='evaluates trained models')
    parser.add_argument('--mname', type=str, default=DEFAULT_NAME, help='name of\
        file which contains models')
    ARGS = parser.parse_args()
    main(**vars(ARGS))

'''
modify gym_envs/quadx_envs/quadx_hover_env.py to take start position and
orientation as arguments


'''
