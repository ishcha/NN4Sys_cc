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
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import heapq
import time
import random
import json
import os
import sys
import inspect
import math
from typing import Callable, Generator, Optional

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from pathlib import Path
from common import sender_obs, config
from common.simple_arg_parse import arg_or_default

MAX_CWND = 5000
MIN_CWND = 4

# MAX_RATE = 1000
MAX_RATE = 1000 * 100
MIN_RATE = 40
# MIN_RATE = 10

REWARD_SCALE = 0.001

MAX_STEPS = 200 # was 400 originally

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'

BYTES_PER_PACKET = 1500

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = False
MAX_LATENCY_NOISE = 1.1

USE_CWND = False

class Link():

    def __init__(self, bandwidth, delay, queue_size, loss_rate):
        self.bw = float(bandwidth)
        self.dl = delay # latency
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw # in seconds

    def get_cur_queue_delay(self, event_time):
        return max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))

    def get_cur_latency(self, event_time):
        return self.dl + self.get_cur_queue_delay(event_time)

    def packet_enters_link(self, event_time):
        if (random.random() < self.lr):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        # print("Extra delay: %f, Current delay: %f, Max delay: %f" % (extra_delay, self.queue_delay, self.max_queue_delay))
        if extra_delay + self.queue_delay > self.max_queue_delay:
            # print("\tDrop!")
            return False
        self.queue_delay += extra_delay
        # print("\tNew delay = %f" % self.queue_delay)
        return True

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

class Network():
    
    def __init__(self, senders, links):
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.queue_initial_packets()

    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False)) 

    def reset(self):
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()

    def get_cur_time(self):
        return self.cur_time

    def run_for_dur(self, dur):
        end_time = self.cur_time + dur
        for sender in self.senders:
            sender.reset_obs()

        while self.cur_time < end_time:
            event_time, sender, event_type, next_hop, cur_latency, dropped = heapq.heappop(self.q)
            #print("Got event %s, to link %d, latency %f at time %f" % (event_type, next_hop, cur_latency, event_time))
            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            push_new_event = False

            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    if dropped:
                        sender.on_packet_lost()
                        #print("Packet lost at time %f" % self.cur_time)
                    else:
                        sender.on_packet_acked(cur_latency)
                        #print("Packet acked at time %f" % self.cur_time)
                else:
                    new_next_hop = next_hop + 1
                    link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            if event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    #print("Packet sent at time %f" % self.cur_time)
                    if sender.can_send_packet():
                        # Keep sending packets (on Sender's side)
                        sender.on_packet_sent()
                        push_new_event = True
                    heapq.heappush(self.q, (self.cur_time + (1.0 / sender.rate), sender, EVENT_TYPE_SEND, 0, 0.0, False))
                
                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1
                
                link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                if USE_LATENCY_NOISE:
                    link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(self.cur_time)
                   
            if push_new_event:
                heapq.heappush(self.q, (new_event_time, sender, new_event_type, new_next_hop, new_latency, new_dropped))

        sender_mi = self.senders[0].get_run_data()
        throughput = sender_mi.get("recv rate")
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")
        bw_cutoff = self.links[0].bw * 0.8
        lat_cutoff = 2.0 * self.links[0].dl * 1.5
        loss_cutoff = 2.0 * self.links[0].lr * 1.5
        #print("thpt %f, bw %f" % (throughput, bw_cutoff))
        #reward = 0 if (loss > 0.1 or throughput < bw_cutoff or latency > lat_cutoff or loss > loss_cutoff) else 1 #
        
        # Super high throughput
        #reward = REWARD_SCALE * (20.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Very high thpt
        reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)
        
        # High thpt
        #reward = REWARD_SCALE * (5.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Low latency
        #reward = REWARD_SCALE * (2.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        #if reward > 857:
        #print("Reward = %f, thpt = %f, lat = %f, loss = %f" % (reward, throughput, latency, loss))
        
        #reward = (throughput / RATE_OBS_SCALE) * np.exp(-1 * (LATENCY_PENALTY * latency / LAT_OBS_SCALE + LOSS_PENALTY * loss))
        return reward * REWARD_SCALE

class Sender():
    
    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10):
        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd

    _next_id = 1
    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta):
        delta *= config.DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def apply_cwnd_delta(self, delta):
        delta *= config.DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if USE_CWND:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate
        if new_rate > MAX_RATE or new_rate < MIN_RATE:
            print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()
        
        #obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

class SimulatedNetworkEnv(gym.Env):
    """
    Args:
        :param max_initial_send_rate_bw_ratio (float): The maximum ratio between the initial sending rate and the link's bandwidth
        :param min_initial_send_rate_bw_ratio (float): The maximum ratio between the initial sending rate and the link's bandwidth
        :param max_steps (int): Maximum number of steps per episode
        :param logs_saving_interval (int): the number of steps between saving log files
        :param logs_path (str): path to the episodes logs
        :param log_observations (bool): Should observations should be logged
        :param shallow_queue (bool): Should a shallow buffer (i.e., small `latency inflation` and `latency ratio`) be used
        :param shallow_queue_eps (float): if shallow_buffer=True - defines the constraints that ensure the shallow buffer
        :param params_generator (Optional[Generator]): Generator of env random parameters (e.g., bandwidth, delay, initial sending rate)
        :param change_bw_mid_epsiode (bool): indicates whether to change the link's bandwidth mid-episode
        :param new_bw (float): the new (mid-episode) bandwidth
        :param random_value_fn Callable[[float, float], float]: A function with argument (low, high), returns a value that was drawn from some specified distribution in the range [low,high]
    """
    def __init__(self,
                 history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                    default="sent latency inflation,"
                          + "latency ratio,"
                          + "send ratio"),
                 min_initial_send_rate_bw_ratio:float=0.3,
                 max_initial_send_rate_bw_ratio:float=1.5,
                 max_steps:int = MAX_STEPS,
                 logs_saving_interval:int=100,
                 logs_path: str ="",
                 log_observations: bool=False,
                 shallow_queue: bool=False,
                 shallow_queue_eps: float=0.0,
                 params_generator: Optional[Generator]=None,
                 change_bw_mid_epsiode: bool =False,
                 new_bw: float= 500,
                 random_value_fn: Callable[[float, float], float]= random.uniform):
        super().__init__()
        self.viewer = None

        self.min_bw, self.max_bw = (100, 500)
        self.min_lat, self.max_lat = (0.05, 0.5)
        self.min_queue, self.max_queue = (0, 8)
        self.min_loss, self.max_loss = (0.0, 0.05)
        self.history_len = history_len
        print("History length: %d" % history_len)
        self.features = features.split(",")
        print("Features: %s" % str(self.features))

        self.links = None
        self.senders = None
        self.min_initial_send_rate_bw_ratio = min_initial_send_rate_bw_ratio
        self.max_initial_send_rate_bw_ratio = max_initial_send_rate_bw_ratio
        self.logs_saving_interval = logs_saving_interval
        self.log_observations = log_observations
        self.shallow_queue = shallow_queue
        self.shallow_buffer_eps = shallow_queue_eps

        self.params_generator = params_generator
        self.random_value_fn: Callable = random_value_fn

        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.max_steps = max_steps
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        self.set_output_folder(logs_path)

        if USE_CWND:
            self.action_space = spaces.Box(np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(np.array([-1e12]), np.array([1e12]), dtype=np.float32)
                   

        self.observation_space = None
        use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.event_record = {"Events":[]}
        self.episodes_run = -1

        self.change_bw_mid_epsiode = change_bw_mid_epsiode
        
        if change_bw_mid_epsiode:
            self.change_step_num = math.floor(self.max_steps / 2)
            self.new_bw = new_bw

    def set_output_folder(self, folder_name: str)-> None:
        self.file_output_path = folder_name+ "./episodes_data/" 
        Path(self.file_output_path).mkdir(parents=True, exist_ok=True)

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        #print(sender_obs)
        return sender_obs

    def _change_bandwidth(self, new_bandwidth: float) -> None:
        print("Episode: {}, changing BW from {} to {}".format(self.steps_taken, self.bw, new_bandwidth))
        
        self.bw = new_bandwidth
        new_links = [Link(self.bw, self.lat, self.queue, self.loss), Link(self.bw, self.lat, self.queue, self.loss)]
        self.links = new_links
        self.net.links = new_links
        self.senders[0].path = [self.links[0], self.links[1]]

    def step(self, actions):
        for i in range(0, 1):#len(actions)):
            #print("Updating rate for sender %d" % i)
            action = actions
            self.senders[i].apply_rate_delta(action[0])
            if USE_CWND:
                self.senders[i].apply_cwnd_delta(action[1])
        #print("Running for %fs" % self.run_dur)
        reward = self.net.run_for_dur(self.run_dur)
        for sender in self.senders:
            sender.record_run()
        self.steps_taken += 1
        
        if self.steps_taken == 1:
            self.event_record["Episode data"]["BW"] = self.bw
            self.event_record["Episode data"]["Link Latency"] = self.lat
            self.event_record["Episode data"]["Link Loss"] = self.loss
            self.event_record["Episode data"]["Queue size"] = self.queue
            self.event_record["Episode data"]["Initial sending rate [BW]"] = self.initial_sr_in_bw_units
            self.event_record["Episode data"]["Initial sending rate [packets]"] = self.initial_sr_in_bw_units * self.bw

        sender_observations = self._get_all_sender_obs()
        sender_mi = self.senders[0].get_run_data()
        event = {}
        event["Name"] = "Step"
        event["Time"] = self.steps_taken
        event["Reward"] = reward
        #event["Target Rate"] = sender_mi.target_rate
        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Avg Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        #event["Cwnd"] = sender_mi.cwnd
        #event["Cwnd Used"] = sender_mi.cwnd_used
        event["Action"] = actions[0].item()
        event["Sending rate"] = self.senders[0].rate
        event["BW"] = self.bw

        if self.log_observations:
            event["Observation"] = list(sender_observations)

        self.event_record["Events"].append(event)
        
        if event["Avg Latency"] > 0.0:
            self.run_dur = 0.5 * sender_mi.get("avg latency")
        #print("Sender obs: %s" % sender_obs)

        should_stop = False

        self.reward_sum += reward

        if self.change_bw_mid_epsiode and (self.steps_taken == self.change_step_num):
            self._change_bandwidth(self.new_bw)
        
        return sender_observations, reward, (self.steps_taken >= self.max_steps or should_stop), {}

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    @staticmethod
    def shallow_queue_size(bw: float, lat: float, eps: float) -> int:
        """
        Args:
            `bw`: the link's bandwidth
            `lat`: 0 the link's latency
            `eps`: a constraint factor (see below)

        Given the arguments calculate the queue size such that:
            1. The `latency ratio` metric is always less or equal to 1+eps
            2. The `latency inflation` (aka latency graident) is always between [-eps, eps]
        """
        # notice that latency_inflation_constraint is always smaller than latency_ratio_constraint, so we just use the smaller one.
        # in general, we would choose queue_size = np.floor(min{latency_ratio_constraint, latency_inflation_constraint})

        # latency_ratio_constraint = 2*eps*bw*lat + 1
        latency_inflation_constraint = eps*bw*lat + 1

        queue_size = math.floor(latency_inflation_constraint)

        return queue_size

    def create_new_links_and_senders(self):
        # queue's units = [packets]
        if self.shallow_queue is True:
            if self.params_generator is not None:
                params = next(self.params_generator)
                
                bw = params[0]
                lat = params[1]
                initial_sr_in_bw_units = params[2]
                loss = params[3]
                assert loss == 0.0
            
            else:
                # Choose bw, lat values from the bigger environment interval (e.g., [0.8*max_bw, max_bw])
                # to enable bigger queue sizes while remaining in the env's training ranges
                bw = self.random_value_fn(self.max_bw*0.8, self.max_bw)
                lat = self.random_value_fn(self.max_lat*0.8, self.max_lat)
                initial_sr_in_bw_units = self.random_value_fn(self.min_initial_send_rate_bw_ratio, self.max_initial_send_rate_bw_ratio)
                loss  = 0.00

            queue_size = SimulatedNetworkEnv.shallow_queue_size(bw, lat, self.shallow_buffer_eps)
            

        else:
            if self.params_generator is not None:
                params = next(self.params_generator)
                
                bw = params[0]
                lat = params[1]
                initial_sr_in_bw_units = params[2]
                loss = params[3]
                # assert loss == 0.0
                queue_size_log = params[4]
                queue_size = 1 + int(np.exp(queue_size_log))
                
            else:
                # Original environment ranges
                bw    = self.random_value_fn(self.min_bw, self.max_bw)
                lat   = self.random_value_fn(self.min_lat, self.max_lat)
                queue_size = 1 + int(np.exp(self.random_value_fn(self.min_queue, self.max_queue)))
                loss  = self.random_value_fn(self.min_loss, self.max_loss)
                initial_sr_in_bw_units = self.random_value_fn(self.min_initial_send_rate_bw_ratio, self.max_initial_send_rate_bw_ratio)
        
        print("BW: {}, latency: {}, Queue size: {}, Init sending rate: {}".format(bw, lat, queue_size, initial_sr_in_bw_units))

        self.links = [Link(bw, lat, queue_size, loss), Link(bw, lat, queue_size, loss)]
        
        self.senders = [Sender(initial_sr_in_bw_units * bw, [self.links[0], self.links[1]], 0, self.features, history_len=self.history_len)]
        
        # Save as properties for logging
        self.bw = bw
        self.lat = lat
        self.queue = queue_size
        self.loss = loss
        self.initial_sr_in_bw_units = initial_sr_in_bw_units

        self.run_dur = 3 * lat # ~ 1.5 RTT

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.episodes_run += 1
        
        if self.episodes_run > 0 and self.episodes_run % self.logs_saving_interval == 0:
            self.dump_events_to_file("pcc_logs/pcc_env_log_run_%d.json" % self.episodes_run)
        
        self.event_record = {"Events":[],
                             "Episode data":{}}

        self.steps_taken = 0
        self.net.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)

        # Removed from the original Aurora code:
        # self.net.run_for_dur(self.run_dur)
        # self.net.run_for_dur(self.run_dur) 
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        print("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0
        return self._get_all_sender_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dump_events_to_file(self, filename):
        filename_path = self.file_output_path + filename
        # with open(filename_path, 'w') as f:
        #     json.dump(self.event_record, f, indent=4)

register(id='PccNs-v0-ood', entry_point='network_sim_ood:SimulatedNetworkEnv')
