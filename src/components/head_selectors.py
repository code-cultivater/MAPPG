#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/29 23:32
# @Author  : Wubing Chen
# @Site    : 
# @File    : head_selectors.py
# @Software: PyCharm

import numpy as np
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.functional import softmax
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}

class EpsilonGreedtHeadelector():
    def __init__(self, args):
        self.args = args

        # Was there so I used it
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_head(self, head_inputs,t_env, test_mode=False):
        '''

        :param head_inputs:[batch_size,n_agent,n_heads]torch.Size([1, 3, 5])
        # not need  :param avail_heads: [batch_size,n_agent,n_heads] marked as 1 or 0
        :param t_env:
        :param test_mode:
        :return: [batch_size,n_agents]
        '''
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        avail_heads=th.ones_like(head_inputs)#torch.Size([1, 3, 5])

        head_values = head_inputs.clone().detach()#[batch_size,n_agent,n_heads] torch.Size([1, 3, 5])


        random_numbers = th.rand_like(head_values[:,:,0])# [batch_size,n_agents] torch.Size([1, 3])
        pick_random = (random_numbers < self.epsilon).long()# [batch_size,n_agents] torch.Size([1, 3])
        random_actions = Categorical(avail_heads).sample().long()#  # [batch_size,n_agents] torch.Size([1, 3])

        picked_actions = pick_random * random_actions + (1 - pick_random) * head_values.max(dim=2)[1]#torch.Size([1, 3])
        return picked_actions

REGISTRY["epsilon_greedy"]=EpsilonGreedtHeadelector