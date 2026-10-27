import minari
import os
import sys
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from gymnasium.spaces import Box, Sequence
from typing import Dict, Union

from stable_baselines3 import A2C, DDPG, DQN, SAC, TD3, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minari import DataCollector
from minari.serialization import deserialize_space, serialize_space

import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv

from gymnasium.envs.registration import EnvSpec

import json
import pickle
import warnings
import multiprocessing as mp
import uuid
import time

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

DEFAULT_ENV = 0
DEFAULT_ALGO = 0

NUM_SAMPLES = int(1e6)

DEFAULT_MAGNITUDE = 1
DEFAULT_SAMPLE_MULT = 1

DEFAULT_LOWER_BOUND = 0
DEFAULT_UPPER_BOUND = 0

DEFAULT_NUM_WORKERS = 4

torch.manual_seed(42)


envs = [
    "PyFlyt/QuadX-Hover-v4",
    "PyFlyt/QuadX-Pole-Balance-v4",
    "PyFlyt/QuadX-Ball-In-Cup-v4",
    "PyFlyt/QuadX-Pole-Waypoints-v4",
    "PyFlyt/QuadX-Waypoints-v4",
    "PyFlyt/Fixedwing-Waypoints-v3",
    "PyFlyt/Rocket-Landing-v4",
]


# Dataset creation
algos = ["a2c", "ddpg", "sac", "td3", "ppo"]

algorithm_map = {
    0: "random",
    10: algos[0],
    11: algos[1],
    12: algos[2],
    13: algos[3],
    14: algos[4],
}


env_config = {
    "render_mode": "rgb_array",
}


# ============================================================
# Minari Sequence serialization
# ============================================================

@serialize_space.register(spaces.Sequence)
def serialize_sequence(
    space: spaces.Sequence,
    to_string=True,
) -> Union[Dict, str]:

    box_obj = space.feature_space

    result = {
        "type": "Sequence",
        "subspaces": serialize_space(box_obj),
    }

    if to_string:
        result = json.dumps(result)

    return result


@deserialize_space.register("Sequence")
def deserialize_sequence(space_dict: Dict) -> spaces.Sequence:

    assert space_dict["type"] == "Sequence"

    box_obj = deserialize_space(space_dict["subspaces"])

    return spaces.Sequence(box_obj, stack=True)


# ============================================================
# Environment creation
# ============================================================

def get_env(env_str, lower_bound, upper_bound):
    """
    Create a regular Gymnasium environment.

    Every worker creates its own environment.
    Minari DataCollector wraps this environment directly.
    """

    if env_str in (envs[3], envs[4], envs[5]):

        env = gym.make(env_str)

        context_length = env_config.get("context_length", 4)

        env = FlattenWaypointEnv(
            env,
            context_length,
        )

        return env

    adaptive_train_flag = True
    pos_orn_flag = 1
    flag = 0
    mode = 2

    # Convert bounds from degrees to radians.
    lower_bound = lower_bound * (3.14 / 180)
    upper_bound = upper_bound * (3.14 / 180)

    flight_dome_size = 150

    Z = 10.0

    start_pos_dict = {
        "x": 0.0,
        "y": 0.0,
        "z": Z,
    }

    start_orn_dict = {
        "r": 0.0,
        "p": 0.0,
        "y": 0.0,
    }

    goal_state_dict = {
        "x": 0.0,
        "y": 0.0,
        "z": Z,
    }

    sparse_reward = 0

    env = gym.make(
        env_str,
        adaptive_train_flag=adaptive_train_flag,
        pos_orn_flag=pos_orn_flag,
        flag=flag,
        mode=mode,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        flight_dome_size=flight_dome_size,
        sparse_reward=sparse_reward,
        max_duration_seconds=10,
        start_pos_dict=start_pos_dict,
        start_orn_dict=start_orn_dict,
        goal_state_dict=goal_state_dict,
    )

    return env


# ============================================================
# Model loading
# ============================================================

def get_model(env, env_str, algorithm_str):

    pack_name, env_name = env_str.split("/")

    model_path = (
        f"../alt_drones/results/dense/"
        f"{env_name}/{algorithm_str}.zip"
    )

    if algorithm_str == "a2c":
        return A2C.load(model_path, env)

    elif algorithm_str == "ddpg":
        return DDPG.load(model_path, env)

    elif algorithm_str == "sac":
        return SAC.load(model_path, env)

    elif algorithm_str == "td3":
        return TD3.load(model_path, env)

    elif algorithm_str == "ppo":
        return PPO.load(model_path, env)

    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm_str}"
        )


# ============================================================
# Policy generation
# ============================================================

def policy_gen(env, algo_num, obs, model=None):

    if algo_num == 0:
        return env.action_space.sample()

    elif 10 <= algo_num <= 14:

        action, _ = model.predict(
            obs,
            deterministic=False,
        )

        return action

    else:
        raise ValueError(
            f"Invalid algorithm number: {algo_num}"
        )


# ============================================================
# Parallel collection worker with progress tracking
# ============================================================

def collect_worker(
    worker_id,
    env_num,
    algo_num,
    target_steps,
    lower_bound,
    upper_bound,
    run_id,
    progress_counter,
    progress_lock,
):
    """
    Collect transitions in an independent worker process.

    Progress is reported through a shared counter, allowing the
    parent process to display one aggregate tqdm progress bar.
    """

    env_str = envs[env_num]
    algorithm_name = algorithm_map[algo_num]

    print(
        f"[Worker {worker_id}] Starting. "
        f"Target: {target_steps:,} transitions.",
        flush=True,
    )

    # Create a regular Gymnasium environment.
    base_env = get_env(
        env_str,
        lower_bound,
        upper_bound,
    )

    # Wrap the environment exactly once.
    collector_env = DataCollector(
        base_env,
        record_infos=True,
    )

    try:

        # Each worker loads its own policy.
        if algo_num == 0:
            model = None

        else:
            model = get_model(
                base_env,
                env_str,
                algorithm_name,
            )

        # Use a different seed for each worker.
        obs, info = collector_env.reset(
            seed=42 + worker_id,
        )

        steps = 0

        while True:

            action = policy_gen(
                collector_env,
                algo_num,
                obs,
                model,
            )

            obs, reward, terminated, truncated, info = (
                collector_env.step(action)
            )

            steps += 1

            # Update the shared progress counter.
            with progress_lock:
                progress_counter.value += 1

            # Finish complete episodes before stopping.
            if terminated or truncated:

                if steps >= target_steps:
                    break

                obs, info = collector_env.reset()

        actual_env_name = env_str.split("/")[-1]

        # Each worker produces a uniquely named dataset.
        shard_id = (
            f"{actual_env_name}/"
            f"worker-{run_id}-{worker_id}-v0"
        )

        collector_env.create_dataset(
            dataset_id=shard_id,
            eval_env=base_env,
            algorithm_name=algorithm_name,
            author="Gideon",
            author_email="filler_email",
            code_permalink=(
                "https://github.com/dangolfecho/alt_offline"
            ),
        )

        print(
            f"[Worker {worker_id}] Finished collecting "
            f"{steps:,} transitions.",
            flush=True,
        )

        print(
            f"[Worker {worker_id}] Saved dataset: {shard_id}",
            flush=True,
        )

        return shard_id

    finally:
        collector_env.close()


# ============================================================
# Main parallel collection
# ============================================================

def main(
    env_num=DEFAULT_ENV,
    algo_num=DEFAULT_ALGO,
    multiplier=DEFAULT_SAMPLE_MULT,
    magnitude=DEFAULT_MAGNITUDE,
    lower_bound=DEFAULT_LOWER_BOUND,
    upper_bound=DEFAULT_UPPER_BOUND,
    num_workers=DEFAULT_NUM_WORKERS,
):

    if not 0 <= env_num < len(envs):
        raise ValueError(
            f"Invalid environment index: {env_num}"
        )

    if algo_num not in algorithm_map:
        raise ValueError(
            f"Invalid algorithm number: {algo_num}"
        )

    if num_workers < 1:
        raise ValueError(
            "num_workers must be at least 1"
        )

    if multiplier < 1 or magnitude < 1:
        raise ValueError(
            "multiplier and magnitude must be positive"
        )

    # Preserve the original number-of-transitions calculation.
    total_steps = (
        multiplier
        * magnitude
        * NUM_SAMPLES
    )

    # Distribute the target transitions among workers.
    steps_per_worker = [
        total_steps // num_workers
        + (
            1
            if worker_id < total_steps % num_workers
            else 0
        )
        for worker_id in range(num_workers)
    ]

    env_str = envs[env_num]
    actual_env_name = env_str.split("/")[-1]
    algorithm_name = algorithm_map[algo_num]

    # Unique ID for this run.
    run_id = uuid.uuid4().hex[:8]

    print("\n" + "=" * 65)
    print("PARALLEL MINARI DATASET COLLECTION")
    print("=" * 65)

    print(f"Environment        : {env_str}")
    print(f"Algorithm          : {algorithm_name}")
    print(f"Number of workers  : {num_workers}")
    print(f"Target transitions : {total_steps:,}")
    print(f"Run ID             : {run_id}")

    print("\nWorker allocation:")

    for worker_id, worker_steps in enumerate(steps_per_worker):

        print(
            f"  Worker {worker_id}: "
            f"{worker_steps:,} transitions"
        )

    print("=" * 65 + "\n")

    # Spawn is safer when initializing independent PyTorch
    # models and physics environments in separate processes.
    ctx = mp.get_context("spawn")

    # A Manager allows worker processes to update shared progress.
    with ctx.Manager() as manager:

        progress_counter = manager.Value("i", 0)
        progress_lock = manager.Lock()

        # Construct arguments for each worker.
        worker_args = [
            (
                worker_id,
                env_num,
                algo_num,
                steps_per_worker[worker_id],
                lower_bound,
                upper_bound,
                run_id,
                progress_counter,
                progress_lock,
            )
            for worker_id in range(num_workers)
        ]

        start_time = time.time()

        # Launch all workers.
        with ctx.Pool(
            processes=num_workers,
        ) as pool:

            result = pool.starmap_async(
                collect_worker,
                worker_args,
            )

            # ------------------------------------------------
            # Aggregate live progress bar
            # ------------------------------------------------

            with tqdm(
                total=total_steps,
                desc="Collecting transitions",
                unit="step",
                dynamic_ncols=True,
                smoothing=0.1,
            ) as pbar:

                last_count = 0

                while not result.ready():

                    current_count = progress_counter.value

                    increment = current_count - last_count

                    if increment > 0:
                        pbar.update(increment)
                        last_count = current_count

                    elapsed = time.time() - start_time

                    if current_count > 0 and elapsed > 0:

                        rate = current_count / elapsed

                        remaining = max(
                            0,
                            total_steps - current_count,
                        )

                        eta = (
                            remaining / rate
                            if rate > 0
                            else 0
                        )

                        pbar.set_postfix({
                            "steps/s": f"{rate:.1f}",
                            "elapsed": f"{elapsed / 60:.1f}m",
                            "ETA": f"{eta / 60:.1f}m",
                        })

                    time.sleep(0.5)

                # Account for the final progress updates.
                current_count = progress_counter.value

                pbar.update(
                    current_count - last_count
                )

                elapsed = time.time() - start_time

                pbar.set_postfix({
                    "elapsed": f"{elapsed / 60:.1f}m",
                    "status": "collection finished",
                })

            # Retrieve worker results and propagate exceptions.
            # This also waits until all workers finish saving
            # their individual Minari datasets.
            shard_ids = result.get()

    print("\nAll workers finished.")
    print("Loading worker datasets...")

    # --------------------------------------------------------
    # Load worker datasets
    # --------------------------------------------------------

    shard_datasets = [
        minari.load_dataset(shard_id)
        for shard_id in shard_ids
    ]

    # Preserve the original dataset naming components.
    final_dataset_id = (
        f"{actual_env_name}/"
        f"dataset-{algo_num}-"
        f"{lower_bound}-{upper_bound}-"
        f"_{magnitude}M-v{multiplier}-{run_id}"
    )

    # --------------------------------------------------------
    # Combine worker datasets
    # --------------------------------------------------------

    print("\nCombining worker datasets...")

    final_dataset = minari.combine_datasets(
        datasets_to_combine=shard_datasets,
        new_dataset_id=final_dataset_id,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DATASET CREATION COMPLETE")
    print("=" * 65)

    print(f"Dataset ID : {final_dataset_id}")
    print(f"Episodes   : {final_dataset.total_episodes:,}")
    print(f"Transitions: {final_dataset.total_steps:,}")

    print("\nWorker datasets:")

    for shard_id in shard_ids:
        print(f"  {shard_id}")

    print("=" * 65 + "\n")

    return final_dataset


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="vector_ppo_collect.py",
        description="Parallel Minari dataset collection",
    )

    # Original positional arguments preserved.
    parser.add_argument(
        "env_num",
        type=int,
        default=DEFAULT_ENV,
        help="which environment to collect data from",
    )

    parser.add_argument(
        "algo_num",
        type=int,
        default=DEFAULT_ALGO,
        help="which algorithm to use for collecting data",
    )

    parser.add_argument(
        "multiplier",
        type=int,
        default=DEFAULT_SAMPLE_MULT,
        help="multiplier for number of samples",
    )

    parser.add_argument(
        "magnitude",
        type=int,
        default=DEFAULT_MAGNITUDE,
        help="how many millions of samples",
    )

    parser.add_argument(
        "lower_bound",
        type=int,
        default=DEFAULT_LOWER_BOUND,
        help="lower bound",
    )

    parser.add_argument(
        "upper_bound",
        type=int,
        default=DEFAULT_UPPER_BOUND,
        help="upper bound",
    )

    # Only new command-line argument.
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="number of parallel workers",
    )

    ARGS = parser.parse_args()

    main(**vars(ARGS))
