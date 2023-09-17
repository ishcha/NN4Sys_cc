import gym
import torch
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common.simple_arg_parse import arg_or_default

K = 10


class CustomNetwork_mid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        y = self.policy_net(features)
        return y


class CustomNetwork_mid_parallel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        x1, x2 = torch.split(features, 30, dim=1)
        y1 = self.policy_net(x1)
        y2 = self.policy_net(x2)
        # Marabou does not support sub
        y2 = torch.mul(y2, -1)
        ret = torch.add(y1, y2)
        return ret


class CustomNetwork_mid_concatnate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, features):
        print(features.size())
        x1, x2, x3, x4, x5, constant = torch.split(features, 30, dim=1)


        y = self.policy_net(x1)
        y = torch.mul(y, 0.025)
        ret = torch.add(y, 1)

        print(x2)

        y1,y2,y3 = torch.split(x2,[19,1,10],dim=1)
        y2 = torch.add(y2,y.detach())
        x2 = torch.concat([y1,y2,y3],dim=1)
        print(x2)

        y = self.policy_net(x2)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y = self.policy_net(x3)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y = self.policy_net(x4)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        y = self.policy_net(x5)
        y = torch.mul(y, 0.025)
        y = torch.add(y, constant)
        ret = ret * y

        return ret


class CustomNetwork_hard(torch.nn.Module):
    def __init__(self, feature_dim: int = K * 3,
                 last_layer_dim_pi: int = 1,
                 last_layer_dim_vf: int = 1):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.conv1 = torch.nn.Conv1d(1, 8, 4)
        self.linear = torch.nn.Linear(216, last_layer_dim_pi)
        self.tanh = torch.nn.Tanh()

        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, last_layer_dim_vf),
            torch.nn.Tanh()
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        x = features.view([-1, 1, 30])
        x = self.conv1(x)
        x = x.view(features.shape[0], -1)
        x = self.linear(x)
        out = self.tanh(x)
        return out

    def forward_critic(self, features):
        return self.value_net(features)


class CustomNetwork_easy(torch.nn.Module):
    def __init__(self, feature_dim: int = K * 3,
                 last_layer_dim_pi: int = 1,
                 last_layer_dim_vf: int = 1):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 16),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, last_layer_dim_pi),
            torch.nn.Tanh()
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, last_layer_dim_vf),
            torch.nn.Tanh()
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)
