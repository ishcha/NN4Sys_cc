import random
import sys

import numpy as np

from pensieve.data_generator import get_inputs_array

MODEL_LIST = ['small', 'mid', 'big']
SPEC_TYPES = ['simple', 'parallel']


def gene_spec():
    for model in MODEL_LIST:
        for spec_type in SPEC_TYPES:
            if spec_type == SPEC_TYPES[0]:
                myarr = np.empty((3000, 48))
            if spec_type == SPEC_TYPES[1]:
                myarr = np.empty((3000, 96))
            for i in range(3000):
                if spec_type == SPEC_TYPES[0]:
                    if i < 15000:
                        # specification 1
                        X = get_inputs_array(0, random_seed=i).flatten()
                    else:
                        # specification 2
                        X = get_inputs_array(1, random_seed=i).flatten()
                else:
                    # specification 3
                    X = get_inputs_array(2, random_seed=i).flatten()
                myarr[i] = X
            np.save(f'./pensieve/pensieve_resources/{model}_{spec_type}.npy', myarr)


def gen_index():
    myarr = np.empty(9000)
    for i in range(3000):
        myarr[i * 3] = 1110000 + i
        myarr[i * 3 + 1] = 3110000 + i
        myarr[i * 3 + 2] = 1120000 + i
    np.save(f'./pensieve/pensieve_resources/pen_difficult.npy', myarr)
    myarr = np.empty(9000)
    for i in range(3000):
        myarr[i * 3] = 1000000 + i
        myarr[i * 3 + 1] = 2000000 + i
        myarr[i * 3 + 2] = 3000000 + i
    np.save(f'./pensieve/pensieve_resources/pen_easy.npy', myarr)
    myarr = np.empty(9000)
    for i in range(3000):
        myarr[i * 3] = 1100000 + i
        myarr[i * 3 + 1] = 2100000 + i
        myarr[i * 3 + 2] = 3100000 + i
    np.save(f'./pensieve/pensieve_resources/pen_medium.npy', myarr)


def main():
    gene_spec()
    gen_index()


if __name__ == "__main__":
    main()
