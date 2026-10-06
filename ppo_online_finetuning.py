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

algos = ["CQL", "IQL"]

dataset_collect_map = {
    0: 'SAC',
    14: 'PPO',
}

def get_dataset_id(
    ac_name: str,
    dataset_algo_num: int,
    dataset_type: int):
    #0 - combined
    #1 - 0-20
    #2 - 20-40
    #3 - 60-80
    if(dataset_type == 0):
        return f'{ac_name}/dataset-{dataset_algo_num}-combined-v0'
    elif(dataset_type == 1):
        return f'{ac_name}/dataset-{dataset_algo_num}-0-20-v3'
    elif(dataset_type == 2):
        return f'{ac_name}/dataset-{dataset_algo_num}-20-40-v2'
    elif(dataset_type == 3):
        return f'{ac_name}/dataset-{dataset_algo_num}-60-80-v1'

def get_model(
    training_algo: int,
    batch_size: int, 
    env,
    save_path,
    device):
    if(training_algo == 0):
        ag = d3rlpy.algos.CQLConfig(batch_size=batch_size).create(device=device)
    elif(training_algo == 1):
        ag = d3rlpy.algos.CQLConfig(batch_size=batch_size).create(device=device)
    ag.build_with_env(env)
    print(save_path)
    ag.load_model(save_path)
    return (ag, algos[training_algo])

DEFAULT_ENV = 0
DEFAULT_DATASET_ALGO = 0
DEFAULT_DTYPE = 0
DEFAULT_TRAINING_ALGO = 0

def main(env_num=DEFAULT_ENV,
    dataset_algo_num=DEFAULT_DATASET_ALGO,
    dataset_type=DEFAULT_DTYPE,
    training_algo=DEFAULT_TRAINING_ALGO
):

    rank = d3rlpy.distributed.init_process_group("gloo")
    print(f"Start running on rank={rank}")

    #device = 'cpu:0'
    device = f'cuda:{rank}'

    pack_name, ac_name = envs[env_num].split('/')
    dataset_id = get_dataset_id(ac_name, dataset_algo_num, dataset_type)
    _, env = d3rlpy.datasets.get_minari(dataset_id,
    action_space=d3rlpy.ActionSpace.CONTINUOUS)

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    batch_size = 512

    save_path = f'models/{dataset_collect_map[dataset_algo]}_{dataset_id}/{algos[training_algo]}/model_2000000.d3'

    ag, algo_str = get_model(training_algo, batch_size, env, save_path, device)

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
    ag.fit_online(env,
            n_steps=int(2e6),
            n_steps_per_epoch=500,
            save_interval=10,
            #logger_adapter=logger_adapter,
            #evaluators={'environment':
                #d3rlpy.metrics.EnvironmentEvaluator(env)},
            #experiment_name=f'Finetune_SAC_{ac_name}_{dataset_num}_SAC',
            experiment_name=f'Online_Finetune_{ac_name}_{dataset_id}_{algo_str}',
            #show_progress=rank == 0,
    )

    ag.save_model(f'models/online_{algo_str}_{ac_name}_{dataset_id}.pt')

    d3rlpy.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='train.py',
            description='does offline training',
            )
    parser.add_argument('env_num', type=int, default=DEFAULT_ENV, help='which\
            environment to train onfrom')
    parser.add_argument('dataset_algo_num', type=int,
    default=DEFAULT_DATASET_ALGO, help='which\
            algorithm which was used for collecting data')
    parser.add_argument('dataset_type', type=int, default=DEFAULT_DTYPE,
    help='which dataset was used for training')
    parser.add_argument('training_algo', type=int,
    default=DEFAULT_TRAINING_ALGO, help='which algo to use for online\
    finetuning')
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
