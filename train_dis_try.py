import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs
import torch
import os

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '4'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]

def main():

    rank = d3rlpy.distributed.init_process_group("gloo")
    print(f"Start running on rank={rank}")

    device = 'cpu:0'


    pack_name, ac_name = envs[0].split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v0')

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    torch.distributed.init_process_group()
    sac = d3rlpy.algos.SACConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            batch_size=256).create(device=device)
    sac = d3rlpy.algos.SACConfig().create()
    #cql = d3rlpy.algos.DiscreteCQLConfig().create()
    logger_adapter: d3rlpy.logging.LoggerAdapterFactory
    evaluators: dict[str, d3rlpy.metrics.EvaluatorProtocol]
    if rank == 0:
        evaluators = {'environment': d3rlpy.metrics.EnvironmentEvaluator(env)}
        logger_adapter = d3rlpy.logging.FileAdapterFactory()
    else:
        evaluators = {}
        logger_adapter = d3rlpy.logging.NoopAdapterFactory()

    #cql.fit(dataset,
    sac.fit(dataset,
            n_steps=int(2e6),
            n_steps_per_epoch=1000,
            save_interval=10,
            logger_adapter=logger_adapter,
            evaluators={'environment':
                d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'SAC_{ac_name}_{0}',
            show_progress=rank == 0,
    )

    d3rlpy.distributed.destroy_process_group()

if __name__ == '__main__':
    main()

'''
Need to do these changes in d3rlpy for this to work

Update
~/miniconda3/envs/env_name/lib/python{version_num}/site-packages/d3rlpy/datasets.py


        if (env.spec.max_episode_steps is None):
            return dataset, GymnasiumTimeLimit(
                unwrapped_env, max_episode_steps=int(1e4)
            )
        else:
            return dataset, GymnasiumTimeLimit(
                unwrapped_env, max_episode_steps=env.spec.max_episode_steps
            )
        and 
def get_minari(
    env_name: str,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    render_mode: Optional[str] = None,
    tuple_observation: bool = False,
    online: bool = True,
) -> tuple[ReplayBuffer, gymnasium.Env[Any, Any]]:

        if(online):
            _dataset = minari.load_dataset(env_name, download=True)
        else:
            _dataset = minari.load_dataset(env_name, download=False)

'''
