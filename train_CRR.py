import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs


DEFAULT_ENV = 'PyFlyt/QuadX-Hover-v4'
DEFAULT_ACTOR_LR = 3e-4
DEFAULT_CRITIC_LR = 3e-4
DEFAULT_BATCH_SIZE = 1024
DEFAULT_GAMMA = 0.99
DEFAULT_BETA = 1.0
DEFAULT_N_ACTION_SAMPLES = 4
DEFAULT_ADVANTAGE_TYPE = 'mean'
DEFAULT_WEIGHT_TYPE = 'exp'
DEFAULT_MAX_WEIGHT = 20.0
DEFAULT_N_CRITICS = 1
DEFAULT_TARGET_UPDATE_TYPE = 'hard'
DEFAULT_TAU = 5e-3
DEFAULT_TARGET_UPDATE_INTERVAL = 100
DEFAULT_UPDATE_ACTOR_INTERVAL = 1
DEFAULT_N_STEPS = int(2e6)
DEFAULT_N_STEPS_PER_EPOCH = int(1e5)
DEFAULT_SAVE_INTERVAL = 10

def main(args):
    pack_name, ac_name = (args.env).split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v0')

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    crr = d3rlpy.algos.CRRConfig(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            beta=args.beta,
            n_action_samples=args.n_action_samples,
            advantage_type=args.advantage_type,
            weight_type=args.weight_type,
            max_weight=args.max_weight,
            n_critics=args.n_critics,
            target_update_type=args.target_update_type,
            tau=args.tau,
            target_update_interval=args.target_update_interval,
            update_actor_interval=args.update_actor_interval,
            ).create()
    
    crr.fit(dataset,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=args.save_interval,
            evaluators={'environment':
                d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'CRR_{ac_name}_{0}',
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--actor_lr', type=float, default=DEFAULT_ACTOR_LR)
    parser.add_argument('--critic_lr', type=float, default=DEFAULT_CRITIC_LR)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA)
    parser.add_argument('--n_action_samples', type=int,
            default=DEFAULT_N_ACTION_SAMPLES)
    parser.add_argument('--advantage_type', type=str,
            default=DEFAULT_ADVANTAGE_TYPE)
    parser.add_argument('--weight_type', type=str,
            default=DEFAULT_WEIGHT_TYPE)
    parser.add_argument('--max_weight', type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument('--n_critics', type=int, default=DEFAULT_N_CRITICS)
    parser.add_argument('--target_update_type', type=str,
            default=DEFAULT_TARGET_UPDATE_TYPE)
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU)
    parser.add_argument('--target_update_interval', type=int,
            default=DEFAULT_TARGET_UPDATE_INTERVAL)
    parser.add_argument('--update_actor_interval', type=int,
            default=DEFAULT_UPDATE_ACTOR_INTERVAL)
    parser.add_argument('--n_steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--n_steps_per_epoch', type=int,
            default=DEFAULT_N_STEPS_PER_EPOCH)
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL)

    args = parser.parse_args()
    main(args)
