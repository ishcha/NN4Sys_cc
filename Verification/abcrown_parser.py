import os
import math
import numpy as np
import sys

def avg_time(dic, times):
    ret={}
    for key in times:
        ret[key] = times[key]/dic[key]
    return ret
def main(model):
    dir = f'./{model}_abcrown_running_result'
    files = os.listdir(dir)

    unsat = 0
    sat = 0
    sat_dic={}
    unsat_dic={}
    sat_time={}
    unsat_time={}
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
                    unsat_time[index] = unsat_time[index] + timeout
                else:
                    unsat_dic[index] = 1
                    unsat_time[index] = timeout
            elif result=='sat':
                print(file)
                sat+=1
                if index in sat_dic.keys():
                    sat_dic[index] = sat_dic[index]+1
                    sat_time[index] = unsat_time[index] + timeout
                else:
                    sat_dic[index] = 1
                    sat_time[index] = timeout
            else:
                print("no result")
                print(file)
    print("----------------------------------------------------------sat files")
    print(f'sat: {sat}')
    print(f'unsat: {unsat}')
    sat_dic = dict(sorted(sat_dic.items()))
    unsat_dic = dict(sorted(unsat_dic.items()))

    sat_avg_time = avg_time(sat_dic, sat_time)
    unsat_avg_time = avg_time(unsat_dic, unsat_time)



    print("sat")

    print(sat_dic)
    print(sat_avg_time)

    print("unsat")

    print(unsat_dic)
    print(unsat_avg_time)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python abcrown_parser.py model")
        exit(1)
    else:
        model = sys.argv[1].lower()

    main(model)
