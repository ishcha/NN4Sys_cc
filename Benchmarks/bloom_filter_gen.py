import sys
import os

import random
import numpy as np
import pandas as pd


P_RANGE = [1/300]
MODELS = ['mid']
DIFFICULTY = ['easy']
SIZES = [10]
SPEC_TYPES = [1]


# responsible for writing the file
def write_vnnlib(X, spec_path):
    with open(spec_path, "w") as f:
        f.write("\n")
        f.write(f"(declare-const X_0 Real)\n")
        f.write(f"(declare-const X_1 Real)\n")
        f.write(f"(declare-const Y_0 Real)\n")
        f.write(
            f"\n(assert (and (>= X_0 {X[0]}) (<= X_0 {X[0] + P_RANGE[0]}) (>= X_1 {X[1]}) (<= X_1 {X[1] + P_RANGE[0]}) (>= Y_0 0.5)))\n")





def write_txt(X, spec_path):
    with open(spec_path, "w") as f:
        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"x{int(i / 2)} >= {X[i]}\n")
            else:
                f.write(f"x{int((i - 1) / 2)} <= {X[i]}\n")

        f.write(f"x0 >= {X[0]}\n")
        f.write(f"x0 <= {X[0] + P_RANGE[0]}\n")
        f.write(f"x1 >= {X[1]}\n")
        f.write(f"x2 <= {X[1] + P_RANGE[0]}\n")
        f.write(f"y0 >= 0.5\n")





def main(random_seed):
    random.seed(random_seed)
    vnn_dir_path = 'vnnlib'
    marabou_txt_dir_path = 'marabou_txt'
    df = pd.read_csv('./src/bloom_filter/bloom_filter_resources/crime_processed.csv')
    all = np.array(df)

    indexes = np.random.randint(0,all.shape[0]-1,size=SIZES[0])
    print(indexes)

    total_num=0

    for index in indexes:
        x=all[index]
        vnn_path = f'{vnn_dir_path}/bloom_filter_{total_num}.vnnlib'
        txt_path = f'{marabou_txt_dir_path}/bloom_filter_{total_num}.txt'
        write_vnnlib(x,vnn_path)
        write_txt(x,txt_path)
        total_num+=1



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_properties.py <random seed>, default is 2024")
        random_seed = 2024
    else:
        random_seed = int(sys.argv[1])
    main(random_seed)