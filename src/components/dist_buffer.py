#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 20:08
# @Author  : Wubing Chen
# @Site    : 
# @File    : dist_buffer.py
# @Software: PyCharm
import numpy as np
class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size,self.n_agents, self.obs_shape]),
                        'actions': np.empty([self.size,self.n_agents, 1]),
                        'state': np.empty([self.size,  self.state_shape]),
                        'reward': np.empty([self.size,  1]),
                        'obs_next': np.empty([self.size,  self.n_agents, self.obs_shape]),
                        'state_next': np.empty([self.size,self.state_shape]),
                        'avail_actions': np.empty([self.size, self.n_agents, self.n_actions]),
                        'avail_actions_next': np.empty([self.size,self.n_agents, self.n_actions]),
                        }



        # store the episode
    def store_batch(self, batch):
        batch_size = batch['obs'].shape[0]  # episode_number

        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = batch['obs']
        self.buffers['actions'][idxs] = batch['actions']
        self.buffers['state'][idxs] = batch['state']
        self.buffers['state_next'][idxs] = batch['state_next']
        self.buffers['reward'][idxs] = batch['reward']
        self.buffers['obs_next'][idxs] = batch['obs_next']

        self.buffers['avail_actions'][idxs] = batch['avail_actions']
        self.buffers['avail_actions_next'][idxs] = batch['avail_actions_next']


    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx