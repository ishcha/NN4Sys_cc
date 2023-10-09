import os
import sys


MODEL_SIZES = ['small', 'mid', 'big']
MODEL_TYPES = ['simple', 'simple', 'simple', 'parallel', 'concat']
running_result_path = './lindex_marabou_running_result'
SIZES = [10, 10, 10, 10, 10]
SIZE = 10
MODEL_NAMES=["lindex","lindex_deep"]



txt_dir_path = '../Benchmarks/marabou_txt'
onnx_dir_path = '../Benchmarks/onnx'


def main(marabou_path):
    if not os.path.exists(running_result_path):
        os.makedirs(running_result_path)
    model_name = "lindex"

    for model in MODEL_NAMES:
        for num in range(SIZE):
            command = f'python {marabou_path} {onnx_dir_path}/{model}.onnx {txt_dir_path}/{model_name}_1_{num}.txt | tee {running_result_path}/{model}_1_{num}.txt'
            print(command)
            os.system(command)
            command = f'python {marabou_path} {onnx_dir_path}/{model}.onnx {txt_dir_path}/{model_name}_2_{num}.txt | tee {running_result_path}/{model}_2_{num}.txt'
            print(command)
            os.system(command)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Lindex_marabou_run.py marabou_path")
        exit(1)
    marabou_path = sys.argv[1]
    main(marabou_path)
