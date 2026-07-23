#!/bin/sh


d3rlpy record d3rlpy_logs/Finetune_SAC_QuadX-Hover-v4_10_SAC_20260721160356/model_100000.d3 \
--env-header 'import PyFlyt.gym_envs; import gymnasium as gym; \
env=gym.make("PyFlyt/QuadX-Hover-v4", render_mode="human")' --out .
