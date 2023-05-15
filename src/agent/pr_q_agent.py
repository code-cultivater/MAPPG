#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 22:32
# @Author  : Wubing Chen
# @Site    : 
# @File    : dist_agent.py
# @Software: PyCharm
import copy

from torch.distributions import Categorical

from components.episode_buffer import EpisodeBatch
from controllers import  REGISTRY as controller_REGISTRY
from learners import  REGISTRY as Learner_REGISTRY
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_attention import QMixerCentralAtten
from modules.mixers.enhanced_qmix import  EnhancedQMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from modules.mixers.enhanced_qmix_central_no_hyper import EnhancedQMixerCentralFF
from modules.mixers.vdn import VDNMixer
import torch as th
import numpy as np
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.critics.coma import COMACritic
from components.action_selectors import REGISTRY as action_REGISTRY
import copy

class PRQAgent:
    def __init__(self, args,buffer,groups,logger):
        self.args=args
        self.critic_ensemble_num=self.args.critic_ensemble_num
        #replay buffer
        self.buffer=buffer



        self.critic_list=[ COMACritic(self.buffer.scheme, args) for i in range(self.critic_ensemble_num)]
        self.target_critic_list=[copy.deepcopy(c) for c in self.critic_list]

        self.action_selector=action_REGISTRY[args.action_selector](self.critic_list,args)
        # controller
        self.controller=controller_REGISTRY[self.args.mac](buffer.scheme, groups, args)
        self.target_control = copy.deepcopy(self.controller)



        # learner
        all_net=[self.controller,self.target_control,self.critic_list,self.target_critic_list]
        self.learner=Learner_REGISTRY[args.learner](self.buffer.scheme, logger,self.args,all_net,self.action_selector)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        '''
        :param ep_batch: 
        :param t_ep: 
        :param t_env: 
        :param bs: 
        :param test_mode: 
        :return: (1,n_agents)
        '''
        if (t_env % 200 < 2 and self.args.name == "pr_q" and t_ep == 0):
            actor_out = self.controller.forward(ep_batch,
                                                t_ep,True)  # (ep_batch.batch_size, self.n_agents, self.args.n_actions)
        else:
            actor_out=self.controller.forward(ep_batch,t_ep)#(ep_batch.batch_size, self.n_agents, self.args.n_actions)
        avaliable_actions=ep_batch["avail_actions"][:, t_ep]
        picked_action=self.action_selector.select_action(actor_out,ep_batch,avaliable_actions,t_ep,t_env,test_mode)


        return  picked_action


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.learner.train(batch, t_env, episode_num)

    def cuda(self):
        self.controller.cuda()
        self.target_control.cuda()
        for i in range(self.critic_ensemble_num):
            self.critic_list[i].cuda()
            self.target_critic_list[i].cuda()


    # TODO: Model saving/loading is out of date!
    def save_models(self, path):
        self.controller.save_models(path)

        #th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    def load_models(self, path):
        self.controller.load_models(path)
        # Not quite right but I don't want to save target networks
        #self.target_central_controller.load_models(path)

    def print_q_table(self,episode_batch):
        self.learner.print_q_table(episode_batch)











