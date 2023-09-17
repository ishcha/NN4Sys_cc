import math

import torch
import torch.nn as nn
import numpy as np

GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
S_INFO = 6
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
cuda = torch.cuda.is_available()
RAND_RANGE = 1000
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = torch.FloatTensor([60.0 * MILLISECONDS_IN_SECOND])  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes


class ActorNetwork_mid(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.conv1 = nn.Conv1d(1, 128, 4)
        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(1, 128)

        self.linear3 = nn.Linear(2048, 128)
        self.linear4 = nn.Linear(128, self.a_dim)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([-1, self.s_dim[0], self.s_dim[1]])
        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        return x


class ActorNetwork_mid_parallel(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.conv1 = nn.Conv1d(1, 128, 4)
        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(1, 128)

        self.linear3 = nn.Linear(2048, 128)
        self.linear4 = nn.Linear(128, self.a_dim)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([-1, 2*self.s_dim[0], self.s_dim[1]])
        x1, x2 = torch.split(x, 6, dim=1)

        split_0 = self.linear0(x1[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x1[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x1[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x1[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x1[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x1[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        out1 = torch.argmax(x, dim=1, keepdim=True)

        split_0 = self.linear0(x2[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x2[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x2[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x2[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x2[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x2[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        out2 = torch.argmax(x, dim=1, keepdim=True)
        return out1 - out2


class ActorNetwork_mid_concat(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.cooked_bw = None
        self.video_size = None
        self.buffer_size = torch.Tensor([0])
        self.video_chunk_sizes = torch.zeros(6)

        self.conv1 = nn.Conv1d(1, 128, 4)
        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(1, 128)

        self.linear3 = nn.Linear(2048, 128)
        self.linear4 = nn.Linear(128, self.a_dim)

    def forward(self, input):
        x = input[:6, :]
        self.cooked_bw = input[6, :]
        video_size = input[7:8, 0]

        self.buffer_size = torch.Tensor([30000])  # fixed
        self.video_chunk_sizes[0] = video_size
        self.video_chunk_sizes[1] = video_size * 2
        self.video_chunk_sizes[2] = video_size * 4
        self.video_chunk_sizes[3] = video_size * 8
        self.video_chunk_sizes[4] = video_size * 16
        self.video_chunk_sizes[5] = video_size * 32

        # step 1
        x = x.view([-1, self.s_dim[0], self.s_dim[1]])
        last_state = x.view([self.s_dim[0], self.s_dim[1]])
        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[0] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 7 / 8

        # step 2
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[1] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 6 / 8

        # step 3
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[2] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 5 / 8

        # step 4
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[3] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 4 / 8

        # step 5
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[4] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 3 / 8

        # step 6
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[5] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 2 / 8

        # step 7
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[6] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 1 / 8

        # step 8
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[7] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 0 / 8

        # Last Prediction
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        return x


class ActorNetwork_small(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(8, 128)
        self.linear3 = nn.Linear(8, 128)
        self.linear4 = nn.Linear(6, 128)
        self.linear5 = nn.Linear(1, 128)

        self.linear6 = nn.Linear(768, 128)
        self.linear7 = nn.Linear(128, self.a_dim)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        return x


class ActorNetwork_small_parallel(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(8, 128)
        self.linear3 = nn.Linear(8, 128)
        self.linear4 = nn.Linear(6, 128)
        self.linear5 = nn.Linear(1, 128)

        self.linear6 = nn.Linear(768, 128)
        self.linear7 = nn.Linear(128, self.a_dim)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([-1, 2*self.s_dim[0], self.s_dim[1]])
        x1, x2 = torch.split(x, 6, dim=1)
        split_0 = self.linear0(x1[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x1[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x1[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x1[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x1[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x1[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        out1 = torch.argmax(x, dim=1, keepdim=True)

        split_0 = self.linear0(x2[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x2[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x2[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x2[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x2[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x2[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        out2 = torch.argmax(x, dim=1, keepdim=True)

        return out1 - out2


class ActorNetwork_small_concat(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.cooked_bw = None
        self.video_size = None
        self.buffer_size = torch.Tensor([0])
        self.video_chunk_sizes = torch.zeros(6)

        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(8, 128)
        self.linear3 = nn.Linear(8, 128)
        self.linear4 = nn.Linear(6, 128)
        self.linear5 = nn.Linear(1, 128)

        self.linear6 = nn.Linear(768, 128)
        self.linear7 = nn.Linear(128, self.a_dim)

    def forward(self, input):
        x = input[:6, :]
        self.cooked_bw = input[6, :]
        video_size = input[7:8, 0]

        self.buffer_size = torch.Tensor([30000])  # fixed
        self.video_chunk_sizes[0] = video_size
        self.video_chunk_sizes[1] = video_size * 2
        self.video_chunk_sizes[2] = video_size * 4
        self.video_chunk_sizes[3] = video_size * 8
        self.video_chunk_sizes[4] = video_size * 16
        self.video_chunk_sizes[5] = video_size * 32

        # step 1
        x = x.view([-1, self.s_dim[0], self.s_dim[1]])
        last_state = x.view([self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[0] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 7 / 8

        # step 2
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[1] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 6 / 8

        # step 3
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[2] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 5 / 8

        # step 4
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[3] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 4 / 8

        # step 5
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[4] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 3 / 8

        # step 6
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[5] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 2 / 8

        # step 7
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[6] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 1 / 8

        # step 8
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[7] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 0 / 8

        # Last Prediction
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.linear2(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.linear3(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.linear4(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear5(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)

        return x


class ActorNetwork_big(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.conv1 = nn.Conv1d(1, 128, 4)
        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(1, 128)

        self.linear3 = nn.Linear(2048, 256)
        self.linear4 = nn.Linear(256, self.a_dim)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])
        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        return x


class ActorNetwork_big_parallel(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        self.conv1 = nn.Conv1d(1, 128, 4)
        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(1, 128)

        self.linear3 = nn.Linear(2048, 256)
        self.linear4 = nn.Linear(256, self.a_dim)

    def forward(self, x):
        # x = torch.reshape(x, (1, self.s_dim[0], self.s_dim[1]))
        x = x.view([-1, 2*self.s_dim[0], self.s_dim[1]])
        x1, x2 = torch.split(x, 6, dim=1)

        split_0 = self.linear0(x1[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x1[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x1[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x1[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x1[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x1[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        out1 = torch.argmax(x, dim=1, keepdim=True)

        split_0 = self.linear0(x2[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x2[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x2[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x2[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x2[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x2[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        out2 = torch.argmax(x, dim=1, keepdim=True)

        return out1 - out2


class ActorNetwork_big_concat(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.cooked_bw = None
        self.video_size = None
        self.buffer_size = torch.Tensor([0])
        self.video_chunk_sizes = torch.zeros(6)

        self.conv1 = nn.Conv1d(1, 128, 4)
        self.relu = nn.ReLU()
        self.linear0 = nn.Linear(1, 128)
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(1, 128)

        self.linear3 = nn.Linear(2048, 256)
        self.linear4 = nn.Linear(256, self.a_dim)

    def forward(self, input):
        input = input.view([-1, self.s_dim[0]+2, self.s_dim[1]])
        x = input[:, :6, :]
        self.cooked_bw = input[:, 6, :]
        video_size = input[:, 7:8, 0]

        self.buffer_size = torch.Tensor([30000])  # fixed
        self.video_chunk_sizes[0] = video_size
        self.video_chunk_sizes[1] = video_size * 2
        self.video_chunk_sizes[2] = video_size * 4
        self.video_chunk_sizes[3] = video_size * 8
        self.video_chunk_sizes[4] = video_size * 16
        self.video_chunk_sizes[5] = video_size * 32

        # step 1
        x = x.view([-1, self.s_dim[0], self.s_dim[1]])
        last_state = x.view([self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        print(self.video_chunk_sizes[action])

        delay = self.video_chunk_sizes[action] / self.cooked_bw[0] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]
        print(video_chunk_size)
        print(delay)

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 7 / 8

        # step 2
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[1] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 6 / 8

        # step 3
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[2] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 5 / 8

        # step 4
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        x = x.view([-1, self.s_dim[0], self.s_dim[1]])
        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[3] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 4 / 8

        # step 5
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[4] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 3 / 8

        # step 6
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[5] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 2 / 8

        # step 7
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[6] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 1 / 8

        # step 8
        last_state = state
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        action = torch.argmax(x, dim=1, keepdim=True).long()
        state = torch.roll(last_state, shifts=-1, dims=1)

        delay = self.video_chunk_sizes[action] / self.cooked_bw[7] - self.buffer_size
        video_chunk_size = self.video_chunk_sizes[action]

        state[0, -1] = VIDEO_BIT_RATE[action] / np.max(VIDEO_BIT_RATE)  # last quality
        state[1, -1] = self.buffer_size / MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
        state[3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = self.video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = 0 / 8

        # Last Prediction
        x = state.view([-1, self.s_dim[0], self.s_dim[1]])

        split_0 = self.linear0(x[:, 0:1, -1])
        split_0 = self.relu(split_0)
        split_1 = self.linear1(x[:, 1:2, -1])
        split_1 = self.relu(split_1)

        split_2 = self.conv1(x[:, 2:3, :])

        split_2 = self.relu(split_2)
        split_3 = self.conv1(x[:, 3:4, :])
        split_3 = self.relu(split_3)
        split_4 = self.conv1(x[:, 4:5, :A_DIM])
        split_4 = self.relu(split_4)
        split_5 = self.linear2(x[:, 4:5, -1])

        split_2 = split_2.view(split_2.shape[0], -1)
        split_3 = split_3.view(split_3.shape[0], -1)
        split_4 = split_4.view(split_4.shape[0], -1)

        x = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 1)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        return x
