import os
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'

MODEL_TYPES = ['simple', 'simple', 'parallel']
MODEL_SIZES = ['small', 'mid', 'big']
P_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SIZES = [5, 5, 5]
SIZE=10
SPEC_TYPES = [1, 2, 3]

# create yaml
vnn_dir_path = '../Benchmarks/vnnlib'
onnx_dir_path = '../Benchmarks/onnx'
yaml_path = './pensieve_yaml'
running_result_path = './pensieve_abcrown_running_result'
timeout = 100
csv_data = []
total_num = 0
current_gpu = 0

if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)


def create_yaml(yaml, vnn_path, onnx_path, inputshape=6):
    with open(yaml, mode='w') as f:
        f.write("general:\n  enable_incomplete_verification: False\n  conv_mode: matrix\n")
        f.write(f'model:\n  onnx_path: {onnx_path}\n')
        f.write('  onnx_quirks: \"{\'Reshape\': {\'fix_batch_size\': True}}\"\n')
        f.write(f'  input_shape: [-1, {inputshape}, 8]\n')
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        f.write(
            "solver:\n  batch_size: 1\nbab:\n  branching:\n    method: sb\n    sb_coeff_thresh: 0.1\n    input_split:\n      enable: True")


def main(abcrown_path):
    for i in range(len(SPEC_TYPES)):
        if i ==0 or i==1:
            continue
        for range_ptr in range(len(P_RANGE)):
            for MODEL in MODEL_SIZES:

                MODEL_TYPE = MODEL_TYPES[i]
                if MODEL_TYPE != 'simple' and range_ptr > 0:
                    continue
                for size in range(SIZE):
                    vnn_path = f'{vnn_dir_path}/pensieve_{SPEC_TYPES[i]}_{range_ptr}_{size}.vnnlib'
                    onnx_path = onnx_dir_path + '/pensieve_' + MODEL + '_' + MODEL_TYPE + '.onnx'
                    yaml = yaml_path + '/pensieve_' + MODEL_TYPE + '-' + MODEL + str(SPEC_TYPES[i]) + '_' + str(
                        size) + '.yaml'
                    if MODEL_TYPE == 'simple':
                        create_yaml(yaml, vnn_path, onnx_path, 6)

                    if MODEL_TYPE == 'parallel':
                        create_yaml(yaml, vnn_path, onnx_path, 12)
                    os.system(
                        f"python {abcrown_path} --config {yaml} | tee {running_result_path}/pensieve_{MODEL}_{SPEC_TYPES[i]}_{range_ptr}_{size}.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Pensieve_abcorwn_run.py abcrown_path")
        exit(1)
    abcrown_path = sys.argv[1]
    main(abcrown_path)
