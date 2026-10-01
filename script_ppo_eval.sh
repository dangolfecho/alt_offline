#!/bin/sh
#env_index dataset_collect_index dataset_index algo_index lower_bound
#upper_bound
python ppo_eval.py 0 1 3 1 0 20 #0-20
python ppo_eval.py 0 1 3 1 20 40 #20-40
python ppo_eval.py 0 1 3 1 40 60 #40-60
python ppo_eval.py 0 1 3 1 60 80 # 60-80
python ppo_eval.py 0 1 3 1 0 80 # full
