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


    for key in datas:
        if 'marabou' in datas[key]:
            weight_counts_marabou['safe'][np_index] = datas[key]['marabou']['safe']
            weight_counts_marabou['unsafe'][np_index] = datas[key]['marabou']['unsafe']
            weight_counts_marabou['timeout'][np_index] = datas[key]['marabou']['timeout']
        if 'abcrown' in datas[key]:
            weight_counts_abcrown['safe'][np_index] = datas[key]['abcrown']['safe']
            weight_counts_abcrown['unsafe'][np_index] = datas[key]['abcrown']['unsafe']
            weight_counts_abcrown['timeout'][np_index] = datas[key]['abcrown']['timeout']
        np_index += 1


    #first figure
    fig, ax = plt.subplots()
    bottom = np.zeros(number)

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 1

    colors = ['#45a776','#3682be','#f05326','#eed777']

    color_ptr=0
    for attribute, weight_count in weight_counts_abcrown.items():
        p = ax.bar(species, weight_count, width, label='alpha-beta-crown:'+attribute, bottom=bottom)
        color_ptr+=1
        bottom += weight_count

    color_ptr = 0
    bottom = np.zeros(number)
    for attribute, weight_count in weight_counts_marabou.items():
        offset = width * multiplier
        p = ax.bar(x+offset, weight_count, width, label='marabou:'+attribute, bottom=bottom)
        color_ptr += 1
        bottom += weight_count
    #ax.legend(ncol=6, bbox_to_anchor=(0, 1), loc="upper left", fontsize='small' )
    ax.legend(loc="upper right", fontsize='small')
    ax.set_title("Number of Solved Cases")
    ax.set_ylabel('Case Number')
    plt.xticks(rotation=-25)
    plt.show()



    #second fiture
    time_sets = {}

    time_sets['abcrown'] = np.zeros(number)
    time_sets['marabou'] = np.zeros(number)
    np_index =0
    for key in datas:
        if 'abcrown' in datas[key]:
            time_sets['abcrown'][np_index] = int(datas[key]['abcrown']['time']*100)/100.0
        if 'marabou' in datas[key]:
            time_sets['marabou'][np_index] = int(datas[key]['marabou']['time']*100)/100.0

        np_index += 1

    x = np.arange(len(species))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in time_sets.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3,fontsize=8)
        multiplier += 1





    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Verification Runtime')
    ax.legend(loc='upper left', ncols=3)
    ax.set_xticks(x + width, species)
    ax.set_ylabel('Time (s)')
    plt.xticks(rotation=-25)

    plt.show()



    # figure 3
    fig, ax = plt.subplots()
    sizes=[]
    runtime=[]
    labels=[]

    size_dic={}
    run_time_dic={}


    for key in datas:
        if 'size' in datas[key]:
            size_dic[key] = datas[key]['size']
            run_time_dic[key]=datas[key]["abcrown"]["time"]
    size_list = sorted(size_dic.items(), key=lambda x:x[1])
    size_dic=dict(size_list)

    for key in size_dic:
        runtime.append(run_time_dic[key])

    print(run_time_dic)

    ax.bar(size_dic.keys(),size_dic.values())
    plt.xticks(rotation=-25)
    ax2 = ax.twinx()
    ax2.plot(size_dic.keys(), runtime,color='r')

    plt.show()






'''

    time_sets = {}
    time_sets['marabou'] = np.zeros((number,10))
    time_sets['abcrown'] = np.zeros((number,10))
    np_index =0
    for key in datas:
        if 'marabou' in datas[key]:
            for i in range(10):
                time_sets['marabou'][np_index][i] = datas[key]['marabou']['timeset'][i]
        if 'abcrown' in datas[key]:
            for i in range(10):
                time_sets['abcrown'][np_index][i] = datas[key]['abcrown']['timeset'][i]
        np_index += 1

    positions = [i*4 for i in range(number)]
    print(len(positions))
    print(time_sets['marabou'].shape)
    fig, ax = plt.subplots()
    positions = [i * 4+0.5 for i in range(number)]
    print(datas.keys())
    VP = ax.boxplot(list(time_sets['abcrown']), positions=positions, widths=1, patch_artist=True,
                    showmeans=False, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                              "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},labels=datas.keys())


    VP = ax.boxplot(list(time_sets['marabou']), positions=positions, widths=1, patch_artist=True,
                    showmeans=False, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                              "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},labels=datas.keys())





    #ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
     #      ylim=(0, 8), yticks=np.arange(1, 8))
    plt.tick_params(axis='x', labelsize=8)
    plt.xticks(rotation=-15)
    ax.set_title('Verification Runtime')

    plt.show()

'''
















if __name__ == "__main__":
    main()