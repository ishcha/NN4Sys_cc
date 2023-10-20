import matplotlib.pyplot as plt
import matplotlib as mpl

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

    '''
        if fig == 0:

        number = len(species)
        weight_counts_abcrown = {}
        weight_counts_abcrown['safe'] = np.zeros(number)
        weight_counts_abcrown['unsafe'] = np.zeros(number)
        weight_counts_abcrown['timeout'] = np.zeros(number)

        weight_counts_marabou = {}
        weight_counts_marabou['safe'] = np.zeros(number)
        weight_counts_marabou['unsafe'] = np.zeros(number)
        weight_counts_marabou['timeout'] = np.zeros(number)

        np_index = 0

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

        # first figure
        fig, ax = plt.subplots()
        bottom = np.zeros(number)

        x = np.arange(len(species))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 1

        colors = ['#45a776', '#3682be', '#f05326', '#eed777']

        color_ptr = 0
        for attribute, weight_count in weight_counts_abcrown.items():
            p = ax.bar(species, weight_count, width, label='alpha-beta-crown:' + attribute, bottom=bottom)
            color_ptr += 1
            bottom += weight_count

        print(species)
        print(weight_counts_marabou)
        color_ptr = 0
        bottom = np.zeros(number)
        for attribute, weight_count in weight_counts_marabou.items():
            offset = width * multiplier
            p = ax.bar(x + offset, weight_count, width, label='marabou:' + attribute, bottom=bottom)
            color_ptr += 1
            bottom += weight_count
        # ax.legend(ncol=6, bbox_to_anchor=(0, 1), loc="upper left", fontsize='small' )
        ax.legend(loc="upper right", fontsize='small')
        ax.set_title("Number of Solved Cases")
        ax.set_ylabel('Case Number')
        plt.xticks(rotation=-25)
        plt.show()
    '''

    # second fiture: runtime

    marabou_can_dic_for_sort = {}
    marabou_cannot_dic_for_sort = {}

    for key in datas:
        if 'marabou' in datas[key]:
            marabou_can_dic_for_sort[key] = datas[key]['abcrown']['time']
        else:
            marabou_cannot_dic_for_sort[key] = datas[key]['abcrown']['time']
    sort = sorted(marabou_can_dic_for_sort.items(), key=lambda x: x[1])
    marabou_can_dic_for_sort = dict(sort)
    sort = sorted(marabou_cannot_dic_for_sort.items(), key=lambda x: x[1])
    marabou_cannot_dic_for_sort = dict(sort)

    time_sets = {}
    annotates = {}

    time_sets['abcrown'] = []
    time_sets['marabou'] = []
    annotates['abcrown'] = []
    annotates['marabou'] = []

    x_ticks = []

    for key in marabou_can_dic_for_sort:
        time_sets['abcrown'].append(int(datas[key]['abcrown']['time'] * 100) / 100.0)
        time_sets['marabou'].append(int(datas[key]['marabou']['time'] * 100) / 100.0)
        annotates['abcrown'].append(int(datas[key]['abcrown']['time'] * 100) / 100.0)
        annotates['marabou'].append(int(datas[key]['marabou']['time'] * 100) / 100.0)
        x_ticks.append(key)

    for key in marabou_cannot_dic_for_sort:
        time_sets['abcrown'].append(int(datas[key]['abcrown']['time'] * 100) / 100.0)
        time_sets['marabou'].append(0)
        annotates['abcrown'].append(int(datas[key]['abcrown']['time'] * 100) / 100.0)
        annotates['marabou'].append('NaN')
        x_ticks.append(key)

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots(layout='constrained', figsize=(18, 6))

    plt.bar(x + width / 2, time_sets['abcrown'], width, label='alpha-beta-crown', edgecolor="black", linewidth=2)

    for i in range(len(x_ticks)):
        plt.text(i + width / 2, time_sets['abcrown'][i], str(annotates['abcrown'][i]), ha='center', va='bottom',
                 fontsize=8)

    ax.bar(x + width / 2 + width, time_sets['marabou'], width, label='marabou', edgecolor="black", linewidth=2)
    for i in range(len(x_ticks)):
        plt.text(i + width / 2 + width, max(time_sets['marabou'][i], 0.013), str(annotates['marabou'][i]),
                 ha='center', va='bottom',
                 fontsize=8)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Verification Runtime')
    ax.legend(loc='upper left', ncols=3)
    # ax.legend(bbox_to_anchor=(1, 0.5), loc=3, borderaxespad=0)
    ax.set_xticks(x + width, x_ticks)
    ax.set_ylabel('Time (s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [x.set_linewidth(2) for x in ax.spines.values()]

    plt.xticks(rotation=-25)
    plt.yscale("log")
    plt.savefig("verification_runtime.pdf", format="pdf", bbox_inches="tight")

    # figure 3
    fig, ax = plt.subplots(layout='constrained', figsize=(18, 6))
    sizes = []
    runtime = []
    labels = []

    size_dic = {}
    run_time_dic = {}

    for key in datas:
        if not "pensieve" in key and not "aurora" in key:
            continue
        if 'size' in datas[key]:
            size_dic[key] = datas[key]['size']
            run_time_dic[key] = datas[key]["abcrown"]["time"]
    size_list = sorted(size_dic.items(), key=lambda x: x[1])
    size_dic = dict(size_list)

    for key in size_dic:
        runtime.append(run_time_dic[key])

    l1 = ax.bar(size_dic.keys(), runtime, label='Verification Runtime', lw=2, ec='black')
    plt.xticks(rotation=-25)
    name = list(size_dic.keys())
    for i in range(len(size_dic.keys())):
        plt.text(name[i], runtime[i], str(int(100 * runtime[i]) / 100.0), ha='center', va='bottom')

    ax2 = ax.twinx()
    size_list = list(size_dic.values())

    l2 = ax2.plot(size_dic.keys(), size_dic.values(), '-o', label='Model Size', color='orange')
    # for i in range(len(size_dic.keys())):
    #    plt.text(name[i], size_list[i], str(int(size_list[i])), ha='center', va='bottom')

    plt.yscale("log")

    ax.set_title('Verification Runtime and Model Size')
    ax2.set_ylabel('Size (byte)')
    ax.set_ylabel('Runtime (s)')
    plt.yscale("log")

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.legend(loc=(0.03, 0.9), ncols=2)
    # plt.rcParams["axes.linewidth"] = 2
    [x.set_linewidth(2) for x in ax.spines.values()]
    plt.savefig("verification_runtime_and_model_size.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
