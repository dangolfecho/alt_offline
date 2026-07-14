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
from stable_baselines3 import A2C, DDPG, DQN, SAC, TD3, PPO
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
algos = ['a2c', 'ddpg', 'sac', 'td3', 'ppo']

algorithm_map = {
        0: 'random',
        10: algos[0],
        11: algos[1],
        12: algos[2], 
        13: algos[3],
        14: algos[4],
        }

def get_model(env, env_str, algorithm_str):
    pack_name, env_name = env_str.split('/')
    #model_path = f'../alt_drones/{env_name}/{algorithm_str}.zip'
    model_path = f'../alt_drones/backup/{algorithm_str}{env_name}.zip'
    if(algorithm_str == 'a2c'):
        return A2C.load(model_path, env)
    elif(algorithm_str == 'ddpg'):
        return DDPG.load(model_path, env)
    elif(algorithm_str == 'sac'):
        return SAC.load(model_path, env)
    elif(algorithm_str == 'td3'):
        return TD3.load(model_path, env)
    elif(algorithm_str == 'ppo'):
        return PPO.load(model_path, env)

def policy_gen(env, algo_num, obs, model=None):
    if(algo_num == 0):
        return env.action_space.sample()
    elif(algo_num >= 10 and algo_num <= 14):
        return model.predict(obs)

def main(env_num=DEFAULT_ENV, algo_num=DEFAULT_ALGO):
    env = gym.make(envs[env_num])
    collector_env = DataCollector(env)

    obs, _ = collector_env.reset()

    if(algo_num == 0):
        model = None
    else:
        model = get_model(env, envs[env_num], algorithm_map[algo_num])

    for _ in range(NUM_SAMPLES):
        if(algo_num == 0):
            action = policy_gen(env, algo_num, obs, model)
        else:
            action, states = policy_gen(env, algo_num, obs, model)
        obs, rew, terminated, truncated, info = collector_env.step(action)

        if terminated or truncated:
            obs, _ = collector_env.reset()
    _, actual_env_name = (envs[env_num]).split('/')
    dataset = collector_env.create_dataset(dataset_id=f'{actual_env_name}/dataset-v{algo_num}',
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
