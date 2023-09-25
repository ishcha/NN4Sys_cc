import os
import math
import numpy as np
import sys

def main(model):
    files = os.listdir(f'./{model}_running_result')

    unsat = 0
    sat = 0
    sat_dic={}
    unsat_dic={}

    for f in files:
        file=f'./{model}_running_result/'+f
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
    print(f'sat: {sat}')
    print(f'unsat: {unsat}')
    print("sat")
    print(sat_dic)
    print("unsat")
    print(unsat_dic)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python abcrown_parser.py model")
        model="aurora"
    else:
        model = sys.argv[1].lower()

    main(model)
