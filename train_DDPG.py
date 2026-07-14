import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs


DEFAULT_ENV = 'PyFlyt/QuadX-Hover-v4'
DEFAULT_BATCH_SIZE = 100
DEFAULT_ACTOR_LR = 3e-4
DEFAULT_CRITIC_LR = 3e-4
DEFAULT_TAU = 0.005
DEFAULT_N_CRITICS = 1
DEFAULT_N_STEPS = int(2e6)
DEFAULT_N_STEPS_PER_EPOCH = int(1e5)
DEFAULT_SAVE_INTERVAL = 10

def main(args):
    pack_name, ac_name = (args.env).split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v0')

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    ddpg = d3rlpy.algos.DDPGConfig(
            batch_size=args.batch_size,
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            tau=args.tau,
            n_critics=args.n_critics
            ).create()
    
    ddpg.fit(dataset,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=args.save_interval,
            evaluators={'environment':
                d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'DDPG_{ac_name}_{0}',
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--actor_lr', type=float, default=DEFAULT_ACTOR_LR)
    parser.add_argument('--critic_lr', type=float, default=DEFAULT_CRITIC_LR)
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU)
    parser.add_argument('--n_critics', type=int, default=DEFAULT_N_CRITICS)
    parser.add_argument('--n_steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--n_steps_per_epoch', type=int,
            default=DEFAULT_N_STEPS_PER_EPOCH)
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL)

    args = parser.parse_args()
    main(args)
