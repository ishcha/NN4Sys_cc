import os
import random
import numpy as np
import sys

dirs=['result/', 'result1/']
files = ["all_dic.npy", "pen_difficult.npy", "pen_easy.npy", "pen_medium.npy"]
all_dic = np.load(files[0])

def issafe(index):
    for i in range(all_dic.shape[0]):
        if all_dic[i,0]==index:
            if all_dic[i][2]>30 and all_dic[i][1]==0:
                print(index)
            if all_dic[i,1]==1:
                return False, None, None
            else:
                return True, all_dic[i][1], all_dic[i][2]

def count():

    all_ptr=0
    all_safe = np.zeros((60000,3))
    for file in files[1:2]:
        total=0
        indexes = np.load(file)
        ret = np.zeros(20000)

        ptr=0

        print(f'{file[:-4]} safe:')
        for i in indexes:
            is_safe, _, timeout = issafe(i)
            if is_safe:
                total+=1
                ret[ptr]=i
                ptr+=1
                all_safe[all_ptr][0] = i
                all_safe[all_ptr][1] = 0
                all_safe[all_ptr][2] = timeout
                if timeout>100:
                    print(all_safe[all_ptr])
                all_ptr+=1
        ret = ret[ret!=0].flatten()
        np.save("safe_"+file,ret)
        print(f'{total}/{indexes.shape[0]}')
        print(np.load("safe_"+file).shape)
    all_safe = all_safe[all_safe.sum(axis=1)!=0,:]
    np.save("safe_"+files[0], all_safe)
    print(np.load("safe_"+files[0]))

def merge():
    all = np.load("safe_" + files[0])
    for i in all:
        if i[2]>30:
            print(i)


def main():
    merge()


if __name__ == "__main__":
    main()
