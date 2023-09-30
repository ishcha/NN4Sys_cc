import os
import sys

MODELS = ['mid', 'mid', 'mid']
MODEL_TYPES=['simple', 'simple','concat']
SIZES = [5,5,5]


SPEC_TYPES = [1,2,3]






def main():
    if not os.path.exists('running_result'):
        os.makedirs('running_result')
    for spec_type_ptr in range(len(SPEC_TYPES)):
        for num in range(SIZES[spec_type_ptr]):
            os.system(f'python resources/runMarabou.py onnx/decima_mid_{MODEL_TYPES[spec_type_ptr]}.onnx marabou_txt/decima_{SPEC_TYPES[spec_type_ptr]}_{num}.txt | tee ./running_result/mid_{MODEL_TYPES[spec_type_ptr]}_{SPEC_TYPES[spec_type_ptr]}_{num}.txt')

if __name__ == "__main__":
    main()