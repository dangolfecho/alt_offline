import d3rlpy
from d3rlpy.metrics.utility import evaluate_qlearning_with_environment
import argparse
import gymnasium as gym
import PyFlyt.gym_envs
import pandas as pd
import numpy as np

DEFAULT_NAME = "PPO_CQL"
DEFAULT_NUM = int(2e6)
DEFAULT_ENV_INDEX = 0
DEFAULT_DATASET_COLLECT_INDEX = 0
DEFAULT_DATASET_INDEX = 0
DEFAULT_ALGO_INDEX = 0
DEFAULT_LOWER_BOUND = 0
DEFAULT_UPPER_BOUND = 0

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]

#dataset-14-x-y-vz_algo.pt

dataset_collect_map = {
    0: 'SAC',
    1: 'PPO',
}

dataset_map = {
    0: "dataset-14-0-20-v3",
    1: "dataset-14-20-40-v2",
    2: "dataset-14-60-80-v1",
    3: "dataset-14-combined-v0",
    }

algo_map = {
    0: "CQL",
    1: "IQL",
}

def get_model(algo_index, env, batch_size, save_path):
    if(algo_index == 0):
        model = d3rlpy.algos.CQLConfig(batch_size=batch_size).create()
        model.build_with_env(env)
        model.load_model(save_path)
    elif(algo_index == 1):
        model = d3rlpy.algos.IQLConfig(batch_size=batch_size).create()
        model.build_with_env(env)
        model.load_model(save_path)
    return model

def main(
        env_index=DEFAULT_ENV_INDEX,
        dataset_collect_index=DEFAULT_DATASET_COLLECT_INDEX,
        dataset_index=DEFAULT_DATASET_INDEX,
        algo_index=DEFAULT_ALGO_INDEX,
        lower_bound=DEFAULT_LOWER_BOUND,
        upper_bound=DEFAULT_UPPER_BOUND,
        ):
    pack_name, ac_name = envs[env_index].split('/')
    #iter_num = DEFAULT_NUM
    #save_path = f'backup/plot_cache/{}/model_' 
    save_path = f'models/{dataset_collect_map[dataset_collect_index]}_{ac_name}/{dataset_map[dataset_index]}_{algo_map[algo_index]}.pt'
    adaptive_train_flag = True
    pos_orn_flag = 1
    flag = 0
    mode = 2
    lower_bound *= (3.14/180)
    upper_bound *= (3.14/180)
    flight_dome_size = 150
    reward_flag = 0
    Z = 10.0
    start_pos_dict = {"x": 0.0, "y": 0.0, "z": Z} 
    start_orn_dict = {"r": 0.0, "p": 0.0, "y": 0.0} 
    goal_state_dict = {"x": 0.0, "y": 0.0, "z": Z} 
    sparse_reward = 0
    env = gym.make(envs[0],
            adaptive_train_flag=adaptive_train_flag,
            pos_orn_flag=pos_orn_flag,
            flag=flag,
            mode=mode,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            flight_dome_size=flight_dome_size,
            sparse_reward=sparse_reward,
            max_duration_seconds=10,
            start_pos_dict=start_pos_dict,
            start_orn_dict=start_orn_dict,
            goal_state_dict=goal_state_dict,
            )
    print(save_path)
    batch_size=512
    model = get_model(algo_index, env, batch_size, save_path)
    N_TRIALS = 1000
    mean_episode_return = evaluate_qlearning_with_environment(model, env,
            n_trials=N_TRIALS, return_rewards_list=True)
    df = pd.DataFrame(mean_episode_return)
    df.to_csv(f'rewards/{dataset_map[dataset_index]}_{algo_map[algo_index]}_{lower_bound}_{upper_bound}.csv',
    header=['Reward'], index_label='Iter_Num')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='eval.py',
        description='evaluates trained models')
    parser.add_argument('env_index', type=int,
        default=DEFAULT_ENV_INDEX,
        help='getting env details from envs array')
    parser.add_argument('dataset_collect_index', type=int,
        default=DEFAULT_DATASET_COLLECT_INDEX,
        help='getting dataset collect algo')
    parser.add_argument('dataset_index', type=int,
        default=DEFAULT_DATASET_INDEX,
        help='getting model name from dataset name')
    parser.add_argument('algo_index', type=int,
        default=DEFAULT_ALGO_INDEX,
        help='getting model name from training algo name')
    parser.add_argument('lower_bound', type=int,
        default=DEFAULT_LOWER_BOUND,
        help='lower bound')
    parser.add_argument('upper_bound', type=int,
        default=DEFAULT_UPPER_BOUND,
        help='upper bound')
    ARGS = parser.parse_args()
    main(**vars(ARGS))

'''
modify d3rlpy to return episode wise reward list 

'''
