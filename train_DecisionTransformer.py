import argparse
import d3rlpy
import gymnasium as gym
import PyFlyt.gym_envs


DEFAULT_ENV = 'PyFlyt/QuadX-Hover-v4'
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_HEADS = 1
DEFAULT_NUM_LAYERS = 3
DEFAULT_ATTN_DROPOUT = 0.1
DEFAULT_RESID_DROPOUT = 0.1
DEFAULT_EMBED_DROPOUT = 0.1
DEFAULT_ACTIVATION_TYPE = 'relu'
#DEFAULT_POSITION_ENCODING_TYPE = PositionEncodingType.SIMPLE
DEFAULT_N_STEPS = int(2e6)
DEFAULT_N_STEPS_PER_EPOCH = int(1e5)
DEFAULT_SAVE_INTERVAL = 10

def main(args):
    pack_name, ac_name = (args.env).split('/')
    dataset, env = d3rlpy.datasets.get_minari(f'{ac_name}/dataset-v0')

    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)

    decisiontransformer = d3rlpy.algos.DecisionTransformerConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            attn_dropout=args.attn_dropout,
            resid_dropout=args.resid_dropout,
            embed_dropout=args.embed_dropout,
            activation_type=args.activation_type,
            ).create()
    
    decisiontransformer.fit(dataset,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=args.save_interval,
            #evaluators={'environment':
            #    d3rlpy.metrics.EnvironmentEvaluator(env)},
            experiment_name=f'DecisionTransformer_{ac_name}_{0}',
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float,
            default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--num_heads', type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--attn_dropout', type=float, default=DEFAULT_ATTN_DROPOUT)
    parser.add_argument('--resid_dropout', type=float, default=DEFAULT_RESID_DROPOUT)
    parser.add_argument('--embed_dropout', type=float,
            default=DEFAULT_EMBED_DROPOUT)
    parser.add_argument('--activation_type', type=str,
            default=DEFAULT_ACTIVATION_TYPE)
    parser.add_argument('--n_steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--n_steps_per_epoch', type=int,
            default=DEFAULT_N_STEPS_PER_EPOCH)
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL)

    args = parser.parse_args()
    main(args)
