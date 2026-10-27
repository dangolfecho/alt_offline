import csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


runs = ['DDPG_CQL', 'DDPG_IQL', 'SAC_CQL', 'SAC_IQL', 'PPO_CQL', 'PPO_IQL', 'PPO_TD3BC',
        'PPO_DT_updated']

def main():
    vals = []
    for j in range(0, 80, 20):
        vals.append((j*3.14/180, (j+20)*3.14/180))
    vals.append((0, 80*3.14/180))
    files = os.listdir('rewards/')
    for i in sorted(files):
        a = pd.read_csv(f'rewards/{i}')
        print(i)
        print(a.describe())

    '''
    fig, ax = plt.subplots(layout='constrained')

    res = ax.grouped_bar(vals, tick_labels=algos, group_spacing=1)
    for container in res.bar_containers:
        ax.bar_label(container, padding=3)

    ax.set_ylabel('Average reward')
    ax.set_title('Comparsion of reward across regions')
    ax.legend(loc='lower right', ncols=3)
    ax.set_ylim(-200, 0)

    #plt.gca().invert_yaxis()
    plt.show()
    '''



if __name__ == '__main__':
    main()
