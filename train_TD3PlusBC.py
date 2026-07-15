import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs


DEFAULT_ENV = 'PyFlyt/QuadX-Hover-v4'
DEFAULT_ACTOR_LR = 3e-4
DEFAULT_CRITIC_LR = 3e-3
DEFAULT_BATCH_SIZE = 100
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005
DEFAULT_N_CRITICS = 2
DEFAULT_TARGET_SMOOTHING_SIGMA = 0.2
DEFAULT_TARGET_SMOOTHING_CLIP = 0.5
DEFAULT_ALPHA = 2.5
DEFAULT_UPDATE_ACTOR_INTERVAL = 2
DEFAULT_N_STEPS = int(2e6)
DEFAULT_N_STEPS_PER_EPOCH = int(1e5)
DEFAULT_SAVE_INTERVAL = 10

def main(args):
    pack_name, ac_name = (args.env).split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v0')

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    td3plusbc = d3rlpy.algos.TD3PlusBCConfig(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            n_critics=args.n_critics,
            target_smoothing_sigma=args.target_smoothing_sigma,
            target_smoothing_clip=args.target_smoothing_clip,
            alpha=args.alpha,
            update_actor_interval=args.update_actor_interval,
            ).create()
    
    td3plusbc.fit(dataset,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=args.save_interval,
            evaluators={'environment':
                d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'TD3PlusBC_{ac_name}_{0}',
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--actor_lr', type=float, default=DEFAULT_ACTOR_LR)
    parser.add_argument('--critic_lr', type=float, default=DEFAULT_CRITIC_LR)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU)
    parser.add_argument('--n_critics', type=int, default=DEFAULT_N_CRITICS)
    parser.add_argument('--target_smoothing_sigma', type=float,
            default=DEFAULT_TARGET_SMOOTHING_SIGMA)
    parser.add_argument('--target_smoothing_clip', type=float,
            default=DEFAULT_TARGET_SMOOTHING_CLIP)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--update_actor_interval', type=int,
            default=DEFAULT_UPDATE_ACTOR_INTERVAL)
    parser.add_argument('--n_steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--n_steps_per_epoch', type=int,
            default=DEFAULT_N_STEPS_PER_EPOCH)
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL)

    args = parser.parse_args()
    main(args)
