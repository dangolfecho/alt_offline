import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]

def main():
    pack_name, ac_name = envs[0].split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v0')

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    sac = d3rlpy.algos.SACConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            batch_size=256).create()
    
    sac.fit(dataset,
            n_steps=100000,
            n_steps_per_epoch=1000,
            save_interval=10,
            evaluators={'environment':
                d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'SAC_{ac_name}_{0}',
    )


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
