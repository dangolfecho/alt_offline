#!/bin/sh

torchrun --nproc_per_node=4 ppo_train.py "0" "14" "1" "1"
torchrun --nproc_per_node=4 ppo_train.py "0" "14" "2" "1"
torchrun --nproc_per_node=4 ppo_train.py "0" "14" "3" "1"
#python -m torch.distributed.run train.py "0" "12"
#python -m torch.distributed.run train.py "0" "13"
#python -m torch.distributed.run train.py "0" "14"
