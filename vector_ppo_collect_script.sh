#!/bin/sh

python vector_ppo_collect.py 0 14 3 10 0 20 --num-workers 16
#python vector_ppo_collect.py 0 14 3 100 0 20
#python ppo_collect.py 0 14 2 20 40
#python ppo_collect.py 0 14 1 60 80
