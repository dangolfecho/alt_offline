import minari
import argparse

DEFAULT_ENV = 0
DEFAULT_ALGO = 0

envs = ["QuadX-Hover-v4", "QuadX-Pole-Balance-v4",
        "QuadX-Ball-In-Cup-v4", "QuadX-Pole-Waypoints-v4",
        "QuadX-Waypoints-v4", "Fixedwing-Waypoints-v3", "Rocket-Landing-v4"]

algos = ['a2c', 'ddpg', 'sac', 'td3', 'ppo']

algorithm_map = {
        0: 'random',
        10: algos[0],
        11: algos[1],
        12: algos[2], 
        13: algos[3],
        14: algos[4],
        }

def main(env_num=DEFAULT_ENV, algo_num=DEFAULT_ALGO):
    datasets = []
    multiplier = 3
    for i in range(0, 80, 20):
        if(i == 40):
            continue
        lower_bound = i
        upper_bound = i+20
        dataset_name_format = f'{envs[env_num]}/dataset-v1{multiplier}'
        #dataset_name_format = f'{envs[env_num]}/dataset-{algo_num}-{lower_bound}-{upper_bound}-v{multiplier}'
        datasets.append(minari.load_dataset(dataset_name_format))
        multiplier -= 1
        print(datasets[0].env_spec)
        break
    return
    combine_dataset = minari.combine_datasets(datasets_to_combine=datasets, 
            new_dataset_id=f'{envs[env_num]}/dataset-{algo_num}-combined-v0')
#python ppo_collect.py 0 14 3 0 20
#python ppo_collect.py 0 14 2 20 40
#python ppo_collect.py 0 14 1 60 80

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='collect.py',
            description='does dataset creation',
            )
    parser.add_argument('env_num', type=int, default=DEFAULT_ENV, help='which\
            environment to collect data from')
    parser.add_argument('algo_num', type=int, default=DEFAULT_ALGO, help='which\
            algorithm to use for collecting data')
    ARGS = parser.parse_args()
    main(**vars(ARGS))
