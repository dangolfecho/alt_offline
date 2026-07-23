#!/bin/sh

python -m torch.distributed.run train.py "3" "0"
