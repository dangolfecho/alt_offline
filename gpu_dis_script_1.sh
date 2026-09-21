#!/bin/sh

torchrun --nproc_per_node=4 evaluate.py "0" "14"
#python -m torch.distributed.run train.py "0" "12"
#python -m torch.distributed.run train.py "0" "13"
#python -m torch.distributed.run train.py "0" "14"
