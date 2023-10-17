import matplotlib.pyplot as plt

import copy
import os
import math
import numpy as np
import sys
import json


def main():
    f = open('eval_results.json')
    content = f.read()
    datas = json.loads(content)
    f.close()

    species = []

    for key in datas:
        species.append(key)



    number = len(species)
    weight_counts_abcrown = {}
    weight_counts_abcrown['safe']=np.zeros(number)
    weight_counts_abcrown['unsafe']=np.zeros(number)
    weight_counts_abcrown['timeout']=np.zeros(number)

    weight_counts_marabou = {}
    weight_counts_marabou['safe']=np.zeros(number)
    weight_counts_marabou['unsafe']=np.zeros(number)
    weight_counts_marabou['timeout']=np.zeros(number)

    np_index=0
    print(datas)

    for key in datas:
        print(key)
        if 'marabou' in datas[key]:
            weight_counts_marabou['safe'][np_index] = datas[key]['marabou']['safe']
            weight_counts_marabou['unsafe'][np_index] = datas[key]['marabou']['unsafe']
            weight_counts_marabou['timeout'][np_index] = datas[key]['marabou']['timeout']
        if 'abcrown' in datas[key]:
            weight_counts_abcrown['safe'][np_index] = datas[key]['abcrown']['safe']
            weight_counts_abcrown['unsafe'][np_index] = datas[key]['abcrown']['unsafe']
            weight_counts_abcrown['timeout'][np_index] = datas[key]['abcrown']['timeout']
        np_index += 1



    fig, ax = plt.subplots()
    bottom = np.zeros(number)

    x = np.arange(len(species))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 1.1

    color_ptr=0

    for attribute, weight_count in weight_counts_abcrown.items():
        p = ax.bar(species, weight_count, width, label=attribute, bottom=bottom)
        color_ptr+=1
        bottom += weight_count


    color_ptr = 0
    bottom = np.zeros(number)
    for attribute, weight_count in weight_counts_marabou.items():
        offset = width * multiplier
        p = ax.bar(x+offset, weight_count, width, label=attribute, bottom=bottom)
        color_ptr += 1
        bottom += weight_count
    ax.legend(loc="upper right")



    ax.set_title("Number of penguins with above average body mass")


    plt.show()





if __name__ == "__main__":
    main()