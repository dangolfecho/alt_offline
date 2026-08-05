import d3rlpy
import argparse
import onnxscript

DEFAULT_NAME = "0"
def main(mname=DEFAULT_NAME):
    save_path = f'backup/plot_cache/{mname}/' 
    for i in range(int(1e4), int(2e6)+1, int(1e4)):
        model = d3rlpy.load_learnable(f'{save_path}model_{i}.d3')
        model.save_policy(f'{save_path}model_{i}.onnx')
        print(i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='convert.py',
            description='changes models to onnx')
    parser.add_argument('mname', type=str, default=DEFAULT_NAME, help='name of\
        file which contains models')
    ARGS = parser.parse_args()
    main(**vars(ARGS))

'''
modify gym_envs/quadx_envs/quadx_hover_env.py to take start position and
orientation as arguments


'''
