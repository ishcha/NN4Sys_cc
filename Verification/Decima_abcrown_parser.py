import os
import math
import numpy as np

def main():
    files = os.listdir('./decima_running_result')

    unsat = 0
    sat = 0
    sat_dic={}
    unsat_dic={}

    for file in files:
        file='./decima_running_result/'+file
        if file[-3:] != 'txt':
            continue
        index = file[:-4].split('_')
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
                if index[2] in unsat_dic.keys():
                    unsat_dic[index[2]] = unsat_dic[index[2]]+1
                else:
                    unsat_dic[index[2]] = 1
            elif result=='sat':
                print(file)
                sat+=1
                if index[2] in sat_dic.keys():
                    sat_dic[index[2]] = sat_dic[index[2]]+1
                else:
                    sat_dic[index[2]] = 1
    print(f'sat: {sat}')
    print(f'unsat: {unsat}')
    print("sat")
    print(sat_dic)
    print("unsat")
    print(unsat_dic)


if __name__ == "__main__":
    main()
