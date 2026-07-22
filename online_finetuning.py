import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs
import torch
import os

#os.environ['RANK'] = '0'
#os.environ['WORLD_SIZE'] = '4'
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '5678'

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]


DEFAULT_ENV = 0
DEFAULT_DATASET = 0
def main(env_num=DEFAULT_ENV, dataset_num=DEFAULT_DATASET):

    #rank = d3rlpy.distributed.init_process_group("gloo")
    #print(f"Start running on rank={rank}")

    device = 'cpu:0'

    pack_name, ac_name = envs[env_num].split('/')
    env = gym.make(envs[env_num])

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    sac = d3rlpy.algos.SACConfig().create()
    sac = d3rlpy.load_learnable("d3rlpy_logs/SAC_QuadX-Hover-v4_10_20260720192943/model_2000000.d3")
    #cql = d3rlpy.algos.DiscreteCQLConfig().create()
    logger_adapter: d3rlpy.logging.LoggerAdapterFactory
    evaluators: dict[str, d3rlpy.metrics.EvaluatorProtocol]
    '''
    if rank == 0:
        evaluators = {'environment': d3rlpy.metrics.EnvironmentEvaluator(env)}
        logger_adapter = d3rlpy.logging.FileAdapterFactory()
    else:
        evaluators = {}
        logger_adapter = d3rlpy.logging.NoopAdapterFactory()
    '''
    #cql.fit(dataset,
    sac.fit_online(env,
            n_steps=int(2e6),
            n_steps_per_epoch=1000,
            save_interval=10,
            #logger_adapter=logger_adapter,
            #evaluators={'environment':
                #d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'Finetune_SAC_{ac_name}_{dataset_num}_SAC',
            #show_progress=rank == 0,
    )

    sac.save_model(f'online_SAC_{ac_name}_{dataset_num}_final.pt')

    #d3rlpy.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='train.py',
            description='does offline training',
            )
    parser.add_argument('env_num', type=int, default=DEFAULT_ENV, help='which\
            environment to train onfrom')
    parser.add_argument('dataset_num', type=int, default=DEFAULT_DATASET, help='which\
            algorithm to use for collecting data')
    ARGS = parser.parse_args()
    main(**vars(ARGS))


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
