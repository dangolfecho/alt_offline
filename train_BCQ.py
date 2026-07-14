import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs


DEFAULT_ENV = 'PyFlyt/QuadX-Hover-v4'
DEFAULT_ACTOR_LR = 1e-4
DEFAULT_CRITIC_LR = 1e-4
DEFAULT_IMITATOR_LR = 1e-4
DEFAULT_BATCH_SIZE = 100
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005
DEFAULT_N_CRITICS = 2
DEFAULT_UPDATE_ACTOR_INTERVAL = 1
DEFAULT_LAM = 0.75
DEFAULT_N_ACTION_SAMPLES = 100
DEFAULT_ACTION_FLEXIBILITY = 0.05
DEFAULT_RL_START_STEP = 0
DEFAULT_BETA = 0.5
DEFAULT_N_STEPS = int(2e6)
DEFAULT_N_STEPS_PER_EPOCH = int(1e5)
DEFAULT_SAVE_INTERVAL = 10

def main(args):
    pack_name, ac_name = (args.env).split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v0')

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    bcq = d3rlpy.algos.BCQConfig(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            imitator_learning_rate=args.imitator_lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            n_critics=args.n_critics,
            update_actor_interval=args.update_actor_interval,
            lam=args.lam,
            n_action_samples=args.n_action_samples,
            action_flexibility=args.action_flexibility,
            rl_start_step=args.rl_start_step,
            beta=args.beta,
            ).create()
    
    bcq.fit(dataset,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=args.save_interval,
            evaluators={'environment':
                d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'BCQ_{ac_name}_{0}',
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--actor_lr', type=float, default=DEFAULT_ACTOR_LR)
    parser.add_argument('--critic_lr', type=float, default=DEFAULT_CRITIC_LR)
    parser.add_argument('--imitator_lr', type=float, default=DEFAULT_IMITATOR_LR)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU)
    parser.add_argument('--n_critics', type=int, default=DEFAULT_N_CRITICS)
    parser.add_argument('--update_actor_interval', type=int, default=DEFAULT_UPDATE_ACTOR_INTERVAL)
    parser.add_argument('--lam', type=float, default=DEFAULT_LAM)
    parser.add_argument('--n_action_samples', type=int,
            default=DEFAULT_N_ACTION_SAMPLES)
    parser.add_argument('--action_flexibility', type=float,
            default=DEFAULT_ACTION_FLEXIBILITY)
    parser.add_argument('--rl_start_step', type=int, default=DEFAULT_RL_START_STEP)
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA)
    parser.add_argument('--n_steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--n_steps_per_epoch', type=int,
            default=DEFAULT_N_STEPS_PER_EPOCH)
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL)

    args = parser.parse_args()
    main(args)
