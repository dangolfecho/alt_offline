import d3rlpy
import argparse
import gymnasium as gym
import PyFlyt.gym_envs
import pandas as pd
import numpy as np

DEFAULT_ALGO = "PPO"
DEFAULT_MODEL = "DecisionTransformer"
DEFAULT_NUM = int(1e5)

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]

def main(algoname=DEFAULT_ALGO, mname=DEFAULT_MODEL):
    save_path = f'model_2000000.d3' 
    env = gym.make(envs[0])
    device='cuda:0'
    model = d3rlpy.load_learnable(save_path, device=device)

    model.fit(
        
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='eval.py',
            description='evaluates trained models')
    parser.add_argument('--algoname', type=str, default=DEFAULT_ALGO, help='name of\
        algo used to train the model')
    parser.add_argument('--mname', type=str, default=DEFAULT_MODEL, help='name of\
        file which contains models')
    ARGS = parser.parse_args()
    main(**vars(ARGS))

'''
modify gym_envs/quadx_envs/quadx_hover_env.py to take start position and
orientation as arguments


'''
