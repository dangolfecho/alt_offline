import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs


DEFAULT_ENV = 'PyFlyt/QuadX-Hover-v4'
DEFAULT_ACTOR_LR = 1e-4
DEFAULT_CRITIC_LR = 3e-4
DEFAULT_IMITATOR_LR = 3e-4
DEFAULT_TEMP_LR = 1e-4
DEFAULT_ALPHA_LR = 1e-3
DEFAULT_BATCH_SIZE = 256
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005
DEFAULT_N_CRITICS = 2
DEFAULT_INITIAL_TEMPERATURE = 1.0
DEFAULT_INITIAL_ALPHA = 1.0
DEFAULT_ALPHA_THRESHOLD = 0.05
DEFAULT_LAM = 0.75
DEFAULT_N_ACTION_SAMPLES = 100
DEFAULT_N_TARGET_SAMPLES = 10
DEFAULT_N_MMD_ACTION_SAMPLES = 4
DEFAULT_MMD_KERNEL = 'laplacian'
DEFAULT_MMD_SIGMA = 20.0
DEFAULT_VAE_KL_WEIGHT = 0.5
DEFAULT_WARMUP_STEPS = 40000
DEFAULT_N_STEPS = int(2e6)
DEFAULT_N_STEPS_PER_EPOCH = int(1e5)
DEFAULT_SAVE_INTERVAL = 10

def main(args):
    pack_name, ac_name = (args.env).split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v0')

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    bear = d3rlpy.algos.BEARConfig(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            imitator_learning_rate=args.imitator_lr,
            temp_learning_rate=args.temp_lr,
            alpha_learning_rate=args.alpha_lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            n_critics=args.n_critics,
            initial_temperature=args.initial_temperature,
            initial_alpha=args.initial_alpha,
            alpha_threshold=args.alpha_threshold,
            lam=args.lam,
            n_action_samples=args.n_action_samples,
            n_target_samples=args.n_target_samples,
            n_mmd_action_samples=args.n_mmd_action_samples,
            mmd_kernel=args.mmd_kernel,
            mmd_sigma=args.mmd_sigma,
            vae_kl_weight=args.vae_kl_weight,
            warmup_steps=args.warmup_steps,
            ).create()
    
    bear.fit(dataset,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=args.save_interval,
            evaluators={'environment':
                d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'BEAR_{ac_name}_{0}',
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--actor_lr', type=float, default=DEFAULT_ACTOR_LR)
    parser.add_argument('--critic_lr', type=float, default=DEFAULT_CRITIC_LR)
    parser.add_argument('--imitator_lr', type=float, default=DEFAULT_IMITATOR_LR)
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
    parser.add_argument('--lam', type=float, default=DEFAULT_LAM)
    parser.add_argument('--n_action_samples', type=int,
            default=DEFAULT_N_ACTION_SAMPLES)
    parser.add_argument('--n_target_samples', type=int,
            default=DEFAULT_N_TARGET_SAMPLES)
    parser.add_argument('--n_mmd_action_samples', type=int,
            default=DEFAULT_N_MMD_ACTION_SAMPLES)
    parser.add_argument('--mmd_kernel', type=str, default=DEFAULT_MMD_KERNEL)
    parser.add_argument('--mmd_sigma', type=float, default=DEFAULT_MMD_SIGMA)
    parser.add_argument('--vae_kl_weight', type=float,
            default=DEFAULT_VAE_KL_WEIGHT)
    parser.add_argument('--warmup_steps', type=int,
            default=DEFAULT_WARMUP_STEPS)
    parser.add_argument('--n_steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--n_steps_per_epoch', type=int,
            default=DEFAULT_N_STEPS_PER_EPOCH)
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL)

    args = parser.parse_args()
    main(args)
