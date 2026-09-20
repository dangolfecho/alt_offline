#!/bin/sh

python ppo_collect.py 0 14 3 0 20
python ppo_collect.py 0 14 2 20 40
python ppo_collect.py 0 14 1 60 80
