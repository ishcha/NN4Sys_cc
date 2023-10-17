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

    colors = ['#45a776','#3682be','#f05326','#eed777']

    color_ptr=0

    for attribute, weight_count in weight_counts_abcrown.items():
        p = ax.barh(species, weight_count, width, label='alpha-beta-crown:'+attribute, left=bottom)
        color_ptr+=1
        bottom += weight_count



    color_ptr = 0
    bottom = np.zeros(number)
    for attribute, weight_count in weight_counts_marabou.items():
        offset = width * multiplier
        p = ax.barh(x+offset, weight_count, width, label='marabou:'+attribute, left=bottom)
        color_ptr += 1
        bottom += weight_count
    ax.legend(ncol=6, bbox_to_anchor=(0, 1), loc="upper left", fontsize='small' )
    ax.set_title("Number of Solved Cases")
    plt.xticks(rotation=-15)
    plt.show()





if __name__ == "__main__":
    main()