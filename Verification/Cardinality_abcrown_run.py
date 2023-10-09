import os
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'

MODELS = ["cardinality_128", "cardinality_128_dual", "cardinality_2048", "cardinality_2048_dual"]

SIZES = [10, 10, 10, 10, 10]
SIZE = 10



model_name = "cardinality"

# create yaml
vnn_dir_path = '../Benchmarks/vnnlib'
onnx_dir_path = '../Benchmarks/onnx'
yaml_path = f'./{model_name}_yaml'
running_result_path = f'./{model_name}_abcrown_running_result'
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
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        f.write(
            "solver:\n  batch_size: 2048\nbab:\n  branching:\n    method: sb\n    sb_coeff_thresh: 0.1\n    input_split:\n      enable: True")


def main(abcrown_path):
    for MODEL in MODELS:
        for size in range(SIZE):
            vnn_path = f'{vnn_dir_path}/{MODEL}_1_{size}.vnnlib'
            onnx_path = f'{onnx_dir_path}/{MODEL}.onnx'
            yaml = f'{yaml_path}/{MODEL}_1_{size}.yaml'
            create_yaml(yaml, vnn_path, onnx_path)
            os.system(f"python {abcrown_path} --config {yaml} | tee {running_result_path}/{MODEL}_1_{size}.txt")

            vnn_path = f'{vnn_dir_path}/{MODEL}_2_{size}.vnnlib'
            yaml = f'{yaml_path}/{MODEL}_2_{size}.yaml'
            create_yaml(yaml, vnn_path, onnx_path)
            os.system(f"python {abcrown_path} --config {yaml} | tee {running_result_path}/{MODEL}_2_{size}.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Cadinality_abcorwn_run.py abcrown_path")
        exit(1)
    abcrown_path = sys.argv[1]
    main(abcrown_path)
