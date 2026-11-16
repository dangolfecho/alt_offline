#!/bin/sh

python ppo_collect.py 0 14 3 1 0 20
#python ppo_collect.py 0 14 3 100 0 20
python ppo_collect.py 0 14 2 1 20 40
python ppo_collect.py 0 14 1 1 60 80
