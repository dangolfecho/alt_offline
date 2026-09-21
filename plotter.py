import csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


runs = ['DDPG_CQL', 'DDPG_IQL', 'SAC_CQL', 'SAC_IQL', 'PPO_CQL', 'PPO_IQL', 'PPO_TD3BC',
        'PPO_DT_updated']

def main():
    last_vals = []
    for i in runs:
        files = os.listdir(f'd3rlpy_logs/server_runs/{i}/')
        fname = files[0]
        save_row = []
        with open(f'd3rlpy_logs/server_runs/{i}/{fname}/environment.csv',
                newline='') as fp:
            reader = csv.reader(fp)
            for row in reader:
                save_row = row
            last_vals.append(float(save_row[-1]))
    print(last_vals)

    algos = ('DDPG', 'SAC', 'PPO')

    vals = {
            'CQL': (last_vals[0], last_vals[2], last_vals[4],),
            'IQL': (last_vals[1], last_vals[3], last_vals[5],),
            'TD3BC' : (0, 0, last_vals[6]),
            'DecisionTransformer': (0, 0, last_vals[7])
    }

    #vals = {'a': (1, 2, 3)}

    fig, ax = plt.subplots(layout='constrained')

    res = ax.grouped_bar(vals, tick_labels=algos, group_spacing=1)
    for container in res.bar_containers:
        ax.bar_label(container, padding=3)

    ax.set_ylabel('Average reward')
    ax.set_title('Comparsion of reward across models and datasets')
    ax.legend(loc='lower right', ncols=3)
    ax.set_ylim(-200, 0)

    #plt.gca().invert_yaxis()
    plt.show()



if __name__ == '__main__':
    main()
