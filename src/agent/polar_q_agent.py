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

from modules.critics.lica import LICACritic
from components.action_selectors import REGISTRY as action_REGISTRY
import copy

class PolarQAgent:
    def __init__(self, args,buffer,groups,logger):
        self.args=args

        #replay buffer
        self.buffer=buffer



        self.critic= LICACritic(self.buffer.scheme, args)
        self.target_critic=copy.deepcopy(self.critic)

        self.action_selector=action_REGISTRY[self.args.action_selector](self.args)
        # controller
        self.controller=controller_REGISTRY[self.args.mac](buffer.scheme, groups, args)
        self.target_control = copy.deepcopy(self.controller)



        # learner
        all_net=[self.controller,self.critic,self.target_critic]
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

        actor_out=self.controller.forward(ep_batch,t_ep)#(bs, n_agents, n_actions)
        avaliable_actions=ep_batch["avail_actions"][:, t_ep]
        picked_action=self.action_selector.select_action(actor_out[bs],avaliable_actions[bs],t_env,test_mode)
        return  picked_action


    def train(self,t_env, episode):
        episode_sample = None
        for _ in range(self.args.critic_training_iters):
            episode_sample = self.buffer.sample(self.args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != self.args.device:
                episode_sample.to(self.args.device)
            self.learner.train_critic(episode_sample, t_env, episode)

            self.learner.train_actor(episode_sample, t_env, episode)

    def cuda(self):
        self.controller.cuda()
        self.critic.cuda()
        self.target_critic.cuda()


    # TODO: Model saving/loading is out of date!
    def save_models(self, path):
        pass;

    def load_models(self, path):
        pass;

    def print_q_table(self,episode_batch):
        self.learner.print_q_table(episode_batch)











