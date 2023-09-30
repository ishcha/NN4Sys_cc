import os
import sys

STATISTIC_RANGE = [0.05, 0.01, 0, 1]
P_RANGE = [1, 2, 3, 4, 5]
MODELS = ['empty', 'small', 'mid', 'big']
MODEL_TYPES=['simple', 'simple', 'simple', 'parallel', 'concat']
DIFFICULTY = ['easy']
SIZES = [10, 10, 10, 10, 10]
SIZE=50

SPEC_TYPES = [101, 102, 2, 3, 4]
SPEC_ARRAY_LENGTH = [30, 30, 30, 60, 150]
SPEC_ARRAY_NUM = 3000
HISTORY = 10




def main():
    if not os.path.exists('running_result'):
        os.makedirs('running_result')
    if len(sys.argv) != 2:
        print("Usage: run.py runMarabou.py path")
        exit(1)
    marabou_path = sys.argv[1]
    for spec_type in [4]:
        for num in range(SIZE):
            os.system(
                f'python {marabou_path} ../onnxs/aurora_mid_{MODEL_TYPES[spec_type]}.onnx aurora_{SPEC_TYPES[spec_type]}_{num}.txt | tee ./running_result/mid_{MODEL_TYPES[spec_type]}_{SPEC_TYPES[spec_type]}_{num}.txt')


if __name__ == "__main__":
    main()