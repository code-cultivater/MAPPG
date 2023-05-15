#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 15:42
# @Author  : Wubing Chen
# @Site    : 
# @File    : dist_w_net.py
# @Software: PyCharm
import torch
from torch import nn
from  torch.nn import  functional as F
import numpy as np
class VNet(nn.Module):
    def __init__(self, args):
        super(VNet, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.linear1 = nn.linear(self.n_agents * self.n_actions, self.args.mixing_embed_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = F.linear(self.args.mixing_embed_dim, 1)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self,  act, states):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)




class LICAVNet(nn.Module):
    def __init__(self, args):
        super(LICAVNet, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.output_type = "q"

        # Set up network layers
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim * self.n_agents * self.n_actions
        self.hid_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.embed_dim, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hid_dim, self.hid_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.hid_dim)

        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                               nn.ReLU(),
                               nn.Linear(self.hid_dim, 1))
        self.bn=nn.BatchNorm1d(1)

    def forward(self, act, states):#act torch.Size([32, 47, 5, 12])
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim)
        action_probs = act.reshape(-1, 1, self.n_agents * self.n_actions) #torch.Size([1504, 1, 60])

        w1 = self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents * self.n_actions, self.hid_dim)#torch.Size([1504, 60, 64])
        b1 = b1.view(-1, 1, self.hid_dim)

        h = torch.relu(torch.bmm(action_probs, w1) + b1)

        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.hid_dim, 1)

        h2 = torch.bmm(h, w_final)

        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        q = h2 + b2

        q = q.view(bs, -1, 1)
        q=F.sigmoid(q)

        #q=F.normalize(q,p=1)

        return  q #torch.Size([32, 47, 1])


