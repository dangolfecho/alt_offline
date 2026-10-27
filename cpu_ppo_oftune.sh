#!/bin/sh

python cpu_ppo_online_finetuning.py "0" "14" "0" "1" "1"
python cpu_ppo_online_finetuning.py "0" "14" "0" "1" "2"
python cpu_ppo_online_finetuning.py "0" "14" "0" "1" "3"
#torchrun --nproc_per_node=4 ppo_online_finetuning.py "0" "14" "1" "1"
#torchrun --nproc_per_node=4 ppo_online_finetuning.py "0" "14" "2" "1"
#torchrun --nproc_per_node=4 ppo_online_finetuning.py "0" "14" "3" "1"
