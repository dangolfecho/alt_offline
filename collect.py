import minari
import os
import sys
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from rl_zoo3.train import train
from stable_baselines3 import PPO
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import minari
from minari import DataCollector
import PyFlyt.gym_envs


DEFAULT_ENV = 0
DEFAULT_ALGO = 0
NUM_SAMPLES = int(1e3)

torch.manual_seed(42)

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]

#Dataset creation
def policy(env, algo_num):
    #using uniformly random policy
    if(algo_num == 0):
        return env.action_space.sample()

algorithm_map = {
        0: 'random',
        }

def main(env_num=DEFAULT_ENV, algo_num=DEFAULT_ALGO):
    env = minari.DataCollector(gym.make(envs[env_num]))

    pack_name, actual_env_name = envs[env_num].split('/')

    env.reset()

    for _ in range(NUM_SAMPLES):
        action = policy(env, algo_num)
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()

    dataset = env.create_dataset(dataset_id=f'{actual_env_name}/dataset-v{algo_num}',
            eval_env=None,
            author='Gideon',
            author_email='filler_email',
            algorithm_name=algorithm_map[algo_num],
            code_permalink='https://github.com/dangolfecho/alt_offline',
            )
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='collect.py',
            description='does dataset creation',
            )
    parser.add_argument('env_num', type=int, default=DEFAULT_ENV, help='which\
            environment to collect data from')
    parser.add_argument('algo_num', type=int, default=DEFAULT_ALGO, help='which\
            algorithm to use for collecting data')
    ARGS = parser.parse_args()
    main(**vars(ARGS))

#using learnt agent
"""
env = DataCollector(gym.make(envs[0]))
path = os.path.abspath('') + '/logs/ppo/CartPole-v1_1/best_model'
agent = PPO.load(path)

total_episodes = 1_000
for i in tqdm(range(total_episodes)):
    obs, _ = env.reset(seed=42)
    while True:
        action, _ = agent.predict(obs)
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = env.create_dataset(
        dataset_id="cartpole/expert-v0",
        algorithm_name="ExpertPolicy",
        code_permalink="https://minari.farama.org/tutorials/behavioral_cloning",
        author="Farama",
        author_email="contact@farama.org"
)
"""
