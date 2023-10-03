import os
import math
import numpy as np
import sys


def main(model):
    dir = f'./{model}_abcrown_running_result'
    files = os.listdir(dir)

    unsat = 0
    sat = 0
    sat_dic={}
    unsat_dic={}
    print("sat files:----------------------------------------------------------")

    for f in files:
        file=f'{dir}/'+f
        if file[-3:] != 'txt':
            continue
        index = '_'.join(f[:-4].split('_')[:-1])
        timeout=-1

        with open(file,'r') as f:
            result=''
            for line in f:
                line = line.strip()
                if line[:6] == "Result":
                    result = line[8:]
                if line[:4] == "Time":
                    timeout = float(line[6:15])
            if timeout==-1:
                continue
            timeout = math.ceil(timeout)
            if result=='unsat':
                unsat+=1
                if index in unsat_dic.keys():
                    unsat_dic[index] = unsat_dic[index]+1
                else:
                    unsat_dic[index] = 1
            elif result=='sat':
                print(file)
                sat+=1
                if index in sat_dic.keys():
                    sat_dic[index] = sat_dic[index]+1
                else:
                    sat_dic[index] = 1
            else:
                print("no result")
                print(file)
    print("----------------------------------------------------------sat files")
    print(f'sat: {sat}')
    print(f'unsat: {unsat}')
    print("sat")
    sat_dic = dict(sorted(sat_dic.items()))
    print(sat_dic)
    print("unsat")
    unsat_dic = dict(sorted(unsat_dic.items()))
    print(unsat_dic)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python abcrown_parser.py model")
        exit(1)
    else:
        model = sys.argv[1].lower()

    main(model)
