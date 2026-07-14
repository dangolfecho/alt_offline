import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs


DEFAULT_ENV = 'PyFlyt/QuadX-Hover-v4'
DEFAULT_ACTOR_LR = 1e-4
DEFAULT_CRITIC_LR = 3e-4
DEFAULT_TEMP_LR = 1e-4
DEFAULT_ALPHA_LR = 1e-4
DEFAULT_BATCH_SIZE = 256
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005
DEFAULT_N_CRITICS = 2
DEFAULT_INITIAL_TEMPERATURE = 1.0
DEFAULT_INITIAL_ALPHA = 1.0
DEFAULT_ALPHA_THRESHOLD = 10.0
DEFAULT_CONSERVATIVE_WEIGHT = 5.0
DEFAULT_N_ACTION_SAMPLES = 100
DEFAULT_SOFT_Q_BACKUP = False
DEFAULT_MAX_Q_BACKUP = False
DEFAULT_N_STEPS = int(2e6)
DEFAULT_N_STEPS_PER_EPOCH = int(1e5)
DEFAULT_SAVE_INTERVAL = 10

def main(args):
    pack_name, ac_name = (args.env).split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v0')

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    cql = d3rlpy.algos.CQLConfig(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            temp_learning_rate=args.temp_lr,
            alpha_learning_rate=args.alpha_lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            n_critics=args.n_critics,
            initial_temperature=args.initial_temperature,
            initial_alpha=args.initial_alpha,
            alpha_threshold=args.alpha_threshold,
            conservative_weight=args.conservative_weight,
            n_action_samples=args.n_action_samples,
            soft_q_backup=args.soft_q_backup,
            max_q_backup=args.max_q_backup,
            ).create()
    
    cql.fit(dataset,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=args.save_interval,
            evaluators={'environment':
                d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'CQL_{ac_name}_{0}',
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--actor_lr', type=float, default=DEFAULT_ACTOR_LR)
    parser.add_argument('--critic_lr', type=float, default=DEFAULT_CRITIC_LR)
    parser.add_argument('--temp_lr', type=float, default=DEFAULT_TEMP_LR)
    parser.add_argument('--alpha_lr', type=float, default=DEFAULT_ALPHA_LR)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU)
    parser.add_argument('--n_critics', type=int, default=DEFAULT_N_CRITICS)
    parser.add_argument('--initial_temperature', type=float,
            default=DEFAULT_INITIAL_TEMPERATURE)
    parser.add_argument('--initial_alpha', type=float,
            default=DEFAULT_INITIAL_ALPHA)
    parser.add_argument('--alpha_threshold', type=float,
            default=DEFAULT_ALPHA_THRESHOLD)
    parser.add_argument('--conservative_weight', type=float,
            default=DEFAULT_CONSERVATIVE_WEIGHT)
    parser.add_argument('--n_action_samples', type=int,
            default=DEFAULT_N_ACTION_SAMPLES)
    parser.add_argument('--soft_q_backup', type=bool,
            default=DEFAULT_SOFT_Q_BACKUP)
    parser.add_argument('--max_q_backup', type=bool,
            default=DEFAULT_MAX_Q_BACKUP)
    parser.add_argument('--n_steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--n_steps_per_epoch', type=int,
            default=DEFAULT_N_STEPS_PER_EPOCH)
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL)

    args = parser.parse_args()
    main(args)
