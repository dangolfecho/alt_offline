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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minari import DataCollector
from minari.serialization import deserialize_space, serialize_space
import PyFlyt.gym_envs

from gymnasium.spaces import Box, Sequence
from typing import Dict, Union

from tqdm import tqdm
import warnings

from PyFlyt.gym_envs import FlattenWaypointEnv


from gymnasium.envs.registration import EnvSpec

warnings.filterwarnings('ignore')


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

env_config = {
        'render_mode': 'rgb_array',
}


def get_env(env_str):
    if(env_str in (envs[3], envs[4], envs[5])):
        env = gym.make(env_str)
        context_length = env_config.get('context_length', 4)
        env = FlattenWaypointEnv(env, context_length)
        return env
    else:
        return gym.make(env_str)
    
def get_model(env, env_str, algorithm_str):
    pack_name, env_name = env_str.split('/')
    #model_path = f'../alt_drones/{env_name}/{algorithm_str}.zip'
    #model_path = f'../alt_drones/backup/{algorithm_str}{env_name}.zip'
    model_path = f'../alt_drones/results/{env_name}/{algorithm_str}.zip'
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

#If sequence contains box
#Sequence(Box(-x, x, (3,), float64), stack=True)
@serialize_space.register(spaces.Sequence)
def serialize_sequence(space: spaces.Sequence, to_string=True) -> Union[Dict, str]:
    box_obj = ((space.__dict__)['feature_space'])
    result = {}
    result["type"] = "Sequence"
    result["subspaces"] = serialize_space(box_obj)

    if to_string:
        result = json.dumps(result)
    return result

@deserialize_space.register("Sequence")
def deserialize_sequence(space_dict: Dict) -> spaces.Sequence:
    assert space_dict["type"] == "Sequence"
    box_obj = deserialize_space(space_dict['subspaces'])
    return spaces.Sequence(box_obj, stack=True)

def main(env_num=DEFAULT_ENV, algo_num=DEFAULT_ALGO):
    env = get_env(envs[env_num])
    print(env.action_space)
    collector_env = DataCollector(env)

    obs, _ = collector_env.reset()

    if(algo_num == 0):
        model = None
    else:
        model = get_model(env, envs[env_num], algorithm_map[algo_num])

    for _ in tqdm(range(NUM_SAMPLES)):
        if(algo_num == 0):
            action = policy_gen(env, algo_num, obs, model)
        else:
            action, states = policy_gen(env, algo_num, obs, model)
        obs, rew, terminated, truncated, info = collector_env.step(action)

        if terminated or truncated:
            obs, _ = collector_env.reset()
    _, actual_env_name = (envs[env_num]).split('/')
    dataset = collector_env.create_dataset(dataset_id=f'{actual_env_name}/dataset-v{algo_num}',
            eval_env=envs[env_num],
            algorithm_name=algorithm_map[algo_num],
            author='Gideon',
            author_email='filler_email',
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
