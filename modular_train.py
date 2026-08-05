import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs
import torch
import os
import minari
from minari import DataCollector
from gymnasium import spaces
from minari.serialization import deserialize_space, serialize_space
from gymnasium.spaces import Box, Sequence
from typing import Dict, Union


from PyFlyt.gym_envs import FlattenWaypointEnv

#os.environ['RANK'] = '0'
#os.environ['WORLD_SIZE'] = '4'
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '5678'

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]

@serialize_space.register(spaces.Sequence)
def serialize_sequence(space: spaces.Sequence, to_string=True) -> Union[Dict, str]:
    box_obj = ((space.__dict__)['feature_space'])
    result = {}
    result["type"] = "Sequence"
    result["subspaces"] = serialize_space(box_obj)

    if to_string:
        result = json.dumps(result)
    return result

@deserialize_space.register("Sequence")
def deserialize_sequence(space_dict: Dict) -> spaces.Sequence:
    assert space_dict["type"] == "Sequence"
    box_obj = deserialize_space(space_dict['subspaces'])
    return spaces.Sequence(box_obj, stack=True)

DEFAULT_ENV = 0
DEFAULT_DATASET = 0
def main(env_num=DEFAULT_ENV, dataset_num=DEFAULT_DATASET):

    rank = d3rlpy.distributed.init_process_group("gloo")
    print(f"Start running on rank={rank}")

    device = 'cpu:0'

    pack_name, ac_name = envs[env_num].split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v{dataset_num}',
            action_space=d3rlpy.ActionSpace.CONTINUOUS)

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

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
        if(env_num >=3 and env_num <=5):
            evaluators = {'environment':
                    d3rlpy.metrics.EnvironmentEvaluator(FlattenWaypointEnv(env,4))}
        else:
            evaluators = {'environment':
                    d3rlpy.metrics.EnvironmentEvaluator(env)}
        logger_adapter = d3rlpy.logging.FileAdapterFactory()
    else:
        evaluators = {}
        logger_adapter = d3rlpy.logging.NoopAdapterFactory()

    #cql.fit(dataset,
    sac.fit(dataset,
            n_steps=int(1e3),
            #n_steps=int(2e6),
            n_steps_per_epoch=1000,
            save_interval=10,
            logger_adapter=logger_adapter,
            evaluators= evaluators,
            experiment_name=f'SAC_{ac_name}_{dataset_num}',
            show_progress=rank == 0,
    )

    sac.save_model(f'SAC_{ac_name}_{dataset_num}_final.pt')

    d3rlpy.distributed.destroy_process_group()

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
'''
gotta make this change in
also add import gymnasium at top
~/miniconda/envs/d4rl/lib/python3.10/site-packages/PyFlyt/gym_envs/utils/flatten_waypoint_env.py
class FlattenWaypointEnv(ObservationWrapper,
        gymnasium.utils.RecordConstructorArgs):
    """FlattenWaypontEnv."""

    def __init__(self, env: Env, context_length=2):
        """__init__.

        Args:
            env (Env): a PyFlyt Waypoints environment.
            context_length: how many waypoints should be included in the flattened observation space.

        """
        super().__init__(env=env)
        gymnasium.utils.RecordConstructorArgs.__init__(self,
                context_length=context_length)
        if not hasattr(env, "waypoints") and not isinstance(
            env.unwrapped.waypoints,  # type: ignore[reportAttributeAccess]
            WaypointHandler,
        ):
            raise AttributeError(
                "Only a waypoints environment can be used with the `FlattenWaypointEnv` wrapper."
            )
        self.context_length = context_length
        self.attitude_shape = env.observation_space["attitude"].shape[0]  # type: ignore [reportGeneralTypeIssues]
        self.target_shape = env.observation_space["target_deltas"].feature_space.shape[  # type: ignore [reportGeneralTypeIssues]
            0
        ]  # type: ignore [reportGeneralTypeIssues]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.attitude_shape + self.target_shape * self.context_length,),
        )

    def observation(self, observation) -> np.ndarray:
        """Flattens an observation from the super env.

        Args:
            observation: a dictionary observation with an "attitude" and "target_deltas" keys.

        """
'''
"""
following changes to d3rlpy/d3rlpy/datasets.py
add option to get_minari( to pass actionspace continuous.
"""
