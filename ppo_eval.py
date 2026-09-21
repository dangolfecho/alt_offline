import d3rlpy
from d3rlpy.metrics.utility import evaluate_qlearning_with_environment
import argparse
import gymnasium as gym
import PyFlyt.gym_envs
import pandas as pd
import numpy as np

DEFAULT_NAME = "PPO_CQL"
DEFAULT_NUM = int(2e6)
DEFAULT_LOWER_BOUND = 0
DEFAULT_UPPER_BOUND = 0

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]

def main(mname=DEFAULT_NAME,
        lower_bound=DEFAULT_LOWER_BOUND,
        upper_bound=DEFAULT_UPPER_BOUND,
        ):
    iter_num = DEFAULT_NUM
    #save_path = f'backup/plot_cache/{mname}/model_' 
    save_path = f'models/{mname}/model_{iter_num}.d3'
    adaptive_train_flag = True
    pos_orn_flag = 1
    flag = 0
    mode = 2
    lower_bound *= (3.14/180)
    upper_bound *= (3.14/180)
    flight_dome_size = 150
    reward_flag = 0
    Z = 10.0
    start_pos = np.array([[0.0, 0.0, Z]])
    start_orn = np.array([[0.0, 0.0, 0.0]])
    goal_state = np.array([0.0, 0.0, Z])
    sparse_reward = 0
    env = gym.make(envs[0],
            adaptive_train_flag=adaptive_train_flag,
            pos_orn_flag=pos_orn_flag,
            flag=flag,
            mode=mode,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            flight_dome_size=flight_dome_size,
            start_pos=start_pos,
            start_orn=start_orn,
            sparse_reward=sparse_reward,
            goal_state=goal_state,
            max_duration_seconds=10
            )
    model = d3rlpy.load_learnable(save_path)
    N_TRIALS = 2
    mean_episode_return = evaluate_qlearning_with_environment(model, env,
            n_trials=N_TRIALS, return_rewards_list=True)
    print(mean_episode_return)
    df = pd.DataFrame(mean_episode_return)
    df.to_csv("rewards.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='eval.py',
            description='evaluates trained models')
    parser.add_argument('mname', type=str, default=DEFAULT_NAME, help='name of\
        file which contains models')
    parser.add_argument('lower_bound', type=int, default=DEFAULT_LOWER_BOUND,
            help='lower bound')
    parser.add_argument('upper_bound', type=int, default=DEFAULT_UPPER_BOUND,
            help='upper bound')
    ARGS = parser.parse_args()
    main(**vars(ARGS))

'''
modify d3rlpy to return episode wise reward list 

'''
