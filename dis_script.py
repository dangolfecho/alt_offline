#!/bin/sh

python -m torch.distributed.run train.py "0" "11"
python -m torch.distributed.run train.py "0" "12"
python -m torch.distributed.run train.py "0" "13"
python -m torch.distributed.run train.py "0" "14"
