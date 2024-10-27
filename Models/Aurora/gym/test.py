# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import network_sim
import torch
import random
import numpy as np
from model import CustomNetwork_mid, CustomNetwork_big, CustomNetwork_small
import argparse
import json
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common.simple_arg_parse import arg_or_default

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
parser.add_argument('-spec_file', type=str, default='/home/isha/nn4sys/spectra/results/ablation/CC/history_4.txt')
parser.add_argument('--intervene', type=int, default=0)
parser.add_argument('--save_df', action='store_true')
args = parser.parse_args()

env = gym.make('PccNs-v0-pantheon')


with open(args.spec_file, 'r') as f:
    spec_file = f.read()
    specs = spec_file.split('-'*50)[:-1]
    specs = [s.strip().split('\n') for s in specs]
    vals = [(s[1].split('output: ')[-1][1:-1].split(',')) for s in specs]
    vals = [[v.strip()[1:-1] for v in val] for val in vals]
    feats = [json.loads(s[0]) for s in specs]



reversed_xmap = {
    i: f'latency_gradient_{10 - i}' if i <= 9
    else f'latency_ratio_{20 - i}' if i <= 19
    else f'sending_ratio_{30 - i}'
    for i in range(30)
}

xmap = {value: key for key, value in reversed_xmap.items()}


def action_2_symbolic(action):
    if action[0].item() < 0:
        return '-'
    elif action[0].item() > 0:
        return '+'
    else:
        return '0'


def symbolic_2_action(action_symb):
    if action_symb == '-':
        return -0.001
    elif action_symb == '0':
        return 0
    else:
        return 0.001


all_possibilities = ['-', '0', '+']

support = 0
interventions = 0


def intervene(state, action, kind=1):
    support = 0
    interventions = 0
    # check for the precondition from the specs
    # state = [latency_grad 10:1] [latency_ratio 10:1] [sending_rat 10:1]
    possibilities = []
    # print(len(feats))
    for i in range(len(feats)):
        check = True
        for k in feats[i].keys():
            v = state[xmap[k]]
            # print(v, feats[i][k][0], feats[i][k][1])
            if v > feats[i][k][1] or v < feats[i][k][0]:
                check = False
        if check:
            possibilities.extend(vals[i])
        # exit()

    if len(possibilities) > 0:
        support += 1

    action_symb = action_2_symbolic(action)
    if kind == 1:  # do not follow spec if already following
        if action_symb in possibilities:
            possibilities = [i for i in all_possibilities if i not in possibilities]
            if len(possibilities) > 0:
                action_symb = np.random.choice(possibilities)   
                action[0] = symbolic_2_action(action_symb)
                interventions += 1

    
       # kind == 2: follow spec if not already following
    elif kind == 2:
        if action_symb not in possibilities and len(possibilities) > 0:
            action_symb = np.random.choice(possibilities)
            action[0] = symbolic_2_action(action_symb)
            interventions += 1
            
    elif kind == 3:  # remove spec and add back with any random output in the spec
        if action_symb in possibilities:
            # possibilities = [i for i in all_possibilities if i != action_symb]
            if len(possibilities) > 0:
                action_symb = np.random.choice(possibilities)   
                action[0] = symbolic_2_action(action_symb)
                interventions += 1

    return action, support, interventions


@torch.no_grad()
def test(model, save_df=False):
    model.eval()
    support = 0
    interventions = 0
    test_scores = []
    states = []
    actions = []
    for j in range(605):

        state, d, test_score = env.reset(), False, 0

        state = state.astype('float32')
        i = 0
        while not d:

            action = model.forward(torch.tensor(state))[0]
            print(action)
            print(state.shape)
            exit()
            states += [state]
            actions += [action]
            if args.intervene > 0:
                action, sup, inter = intervene(state, action, kind=args.intervene)
                support += sup
                interventions += inter
            state, r, d, _ = env.step(action)
            # print(i, ': step taken: ', d)
            i += 1
            state = state.astype('float32')
            test_score += r
        test_scores.append(test_score)
    if save_df:
        df = pd.DataFrame(states, columns=xmap.keys())
        df['actions'] = actions
        df.to_csv('aurora_df_small_train.csv')
    print('support:', support, 'interventions:', interventions)
    return test_scores


random.seed(0)
np.random.seed(0)


best = 0
best_reward = -100
reward_list = []
for i in range(1):
    model_path = args.model_path
    model = CustomNetwork_mid()

    state_dict = torch.load(model_path)

    for key in list(state_dict.keys()):
        state_dict[key.replace('mlp_extractor.', '')] = state_dict.pop(key)

    state_dict.requires_grad = False
    model.load_state_dict(state_dict, strict=False)
    
    reward2 = test(model, args.save_df)
    # reward1 = test(model)
    sum_tensor = sum(reward2)
    sum_tensor_double = sum_tensor.double()/len(reward2)
    print(sum_tensor_double)
    reward_list.append(sum_tensor_double)
    if sum_tensor_double>best_reward:
        best = i
        best_reward = sum_tensor_double

print(reward_list)
print(f"best: {best}")