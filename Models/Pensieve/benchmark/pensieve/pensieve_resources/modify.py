import os
import random
import numpy as np
import sys

dirs=['result/', 'result1/']
files = ["pen_easy_dic.npy",  "pen_medium_dic.npy", "pen_difficult_dic.npy"]
addtime = [2,3,5]


def addtimes(file, time):
    all = np.load(file)

    for i in range(all.shape[0]):
        print(all[i])
        all[i,2] = all[i,2]+time
        print(all[i])
        print("====")
    np.save(file, all)



def main():
    print(np.load("pen_difficult.npy").shape)
    return
    all = np.load("pen_easy_dic.npy")
    for i in range(all.shape[0]):
        if(all[i,2]>10):
            print(all[i,2])
    return
    np.save("pen_easy_dic.npy", all)
    print(np.load("pen_easy_dic.npy"))
    return
    all = np.load("pen_medium_dic.npy")
    for i in range(all.shape[0]):
        all[i, 2] = 60
    np.save("pen_medium_dic.npy", all)
    print(np.load("pen_medium_dic.npy"))

    all = np.load("pen_difficult_dic.npy")
    for i in range(all.shape[0]):
        all[i, 2] = 140
    np.save("pen_difficult_dic.npy", all)
    print(np.load("pen_difficult_dic.npy"))







if __name__ == "__main__":
    main()