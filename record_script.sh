#!/bin/sh


d3rlpy record d3rlpy_logs/SAC_QuadX-Hover-v4_0_20260718142220/model_2000000.d3 \
--env-header 'import PyFlyt.gym_envs; import gymnasium as gym; \
env=gym.make("PyFlyt/QuadX-Hover-v4", render_mode="human")' --out .
