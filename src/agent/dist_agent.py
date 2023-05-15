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

class DistAgent:
    def __init__(self, args,buffer,groups,):
        self.args=args


        #replay buffer
        self.buffer=buffer
        # controller
        self.controller=controller_REGISTRY[self.args.mac](buffer.scheme, groups, args)
        self.target_control = copy.deepcopy(self.controller)
        # mixer
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer=="enhanced_qmix":
                self.mixer=EnhancedQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.target_mixer = copy.deepcopy(self.mixer)
        #central controller and centrol mixer
        self.central_controller = None

        if self.args.central_mixer in ["ff","enhanced_ff", "atten"]:
            if self.args.central_loss == 0:
                self.central_mixer = self.mixer
                self.central_controller = self.controller
                self.target_central_controller = self.target_control
            else:
                if self.args.central_mixer == "ff":
                    self.central_mixer = QMixerCentralFF(
                        args)  # Feedforward network that takes state and agent utils as input
                elif(self.args.central_mixer == "enhanced_ff"):
                    self.central_mixer = EnhancedQMixerCentralFF(args)
                elif self.args.central_mixer == "atten":
                    self.central_mixer = QMixerCentralAtten(args)
                else:
                    raise Exception("Error with central_mixer")
            assert args.central_mac == "basic_central_mac"
            self.central_controller = controller_REGISTRY[args.central_mac](buffer.scheme,args)
            self.target_central_controller = copy.deepcopy(self.central_controller)
            self.target_central_mixer = copy.deepcopy(self.central_mixer)

        # learner
        all_net=[self.controller,self.target_control,self.mixer,self.target_mixer,self.central_controller,self.target_central_controller,self.central_mixer,self.target_central_mixer]
        self.learner=Learner_REGISTRY[args.learner](self.buffer.scheme, logger,self.args,all_net)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        '''
        :param ep_batch: 
        :param t_ep: 
        :param t_env: 
        :param bs: 
        :param test_mode: 
        :return: (1,n_agents)
        '''
        controller_outputs,chosen_actions=self.controller.get_controller_output(ep_batch,t_ep,t_env,bs,test_mode)#net_outputs  [ k:(batch_size, self.n_agents, n_actions)
        # chosen_actions [ k:(batch_size, self.n_agents)]


        if(test_mode):
            controller_outputs_for_exploit=controller_outputs[: self.args.exploit_agent_num]
            # # average
            # controller_outputs_for_exploit_stack = th.stack(controller_outputs_for_exploit,
            #                                                 dim=controller_outputs_for_exploit[
            #                                                     0].dim())  # (batch_size, self.n_agents, n_actions，k)
            # controller_outputs_for_exploit_sum = controller_oucontroller_outputs_for_exploit_sumtputs_for_exploit_stack.sum(
            #     dim=-1)  # (batch_size, self.n_agents, n_actions)
            # actions = action_selector.select_action(, ep_batch["avail_actions"][:, t_ep][bs], t_env, test_mode = True)
            normal_controller_outputs_for_exploit=[]
            for i in range(self.args.exploit_agent_num):
                controller_outputs_for_exploit_i=controller_outputs_for_exploit[i]

                controller_outputs_for_exploit_i_detach=controller_outputs_for_exploit_i.clone().detach()
                item_num,n_agents,n_actions=controller_outputs_for_exploit_i_detach.size()
                controller_outputs_for_exploit_i_action_max=th.full([item_num,n_agents,1],1.0).to(self.args.device)
                controller_outputs_for_exploit_i_action_min=th.full([item_num,n_agents,1],0.0).to(self.args.device)
                controller_outputs_for_exploit_i_detach_np=controller_outputs_for_exploit_i_detach.cpu().numpy()
                controller_outputs_for_exploit_i_detach_np_ma=np.ma.array(controller_outputs_for_exploit_i_detach_np,mask=1-ep_batch["avail_actions"][:, t_ep][bs].clone().detach().cpu().numpy())
                controller_outputs_for_exploit_i_detach_np_ma_max=np.max(controller_outputs_for_exploit_i_detach_np_ma,axis=-1,keepdims=True)
                controller_outputs_for_exploit_i_detach_np_ma_min = np.min(controller_outputs_for_exploit_i_detach_np_ma, axis=-1, keepdims=True)
                controller_outputs_for_exploit_i_detach_np_max=np.where(controller_outputs_for_exploit_i_detach_np_ma_max==np.ma.masked,1,controller_outputs_for_exploit_i_detach_np_ma_max)
                controller_outputs_for_exploit_i_detach_np_min=np.where(controller_outputs_for_exploit_i_detach_np_ma_min==np.ma.masked,1,controller_outputs_for_exploit_i_detach_np_ma_min)
                normal_controller_outputs_for_exploit_i=(controller_outputs_for_exploit_i_detach-th.from_numpy(controller_outputs_for_exploit_i_detach_np_min).to(self.args.device))/\
                (th.from_numpy(controller_outputs_for_exploit_i_detach_np_max).to(self.args.device)-th.from_numpy(controller_outputs_for_exploit_i_detach_np_min).to(self.args.device))
                normal_controller_outputs_for_exploit_i_detach=normal_controller_outputs_for_exploit_i.clone().detach()
                normal_controller_outputs_for_exploit.append(normal_controller_outputs_for_exploit_i_detach)
            action_selector = action_REGISTRY[self.args.action_selector](self.args)

            actions=None
            if(self.args.test_action=="average"):
                # average
                controller_outputs_for_exploit_stack = th.stack(normal_controller_outputs_for_exploit,
                                                            dim=controller_outputs_for_exploit[
                                                                0].dim())  # (batch_size, self.n_agents, n_actions，k)
                controller_outputs_for_exploit_sum = controller_outputs_for_exploit_stack.sum(
                dim=-1)  # (batch_size, self.n_agents, n_actions)
                actions = action_selector.select_action(controller_outputs_for_exploit_sum, ep_batch["avail_actions"][:, t_ep][bs], t_env, test_mode = True)
            elif(self.args.test_action=="single"):
                actions = action_selector.select_action(normal_controller_outputs_for_exploit[0], ep_batch["avail_actions"][:, t_ep][bs], t_env, test_mode = True)
            else:
                raise Exception("not implementation in other test_action")

            return  actions
        else:
            #collect sample
            if(self.args.train_action =="argmax"):
                mixer_outputs=[]
                for k in range(self.args.K):
                    selected_controller_outputs_k=th.gather(controller_outputs[k],dim=2,index=chosen_actions[k].unsqueeze(chosen_actions[k].dim())).squeeze(-1)#(batch_size, self.n_agents)
                    state_t=ep_batch["state"][:,t_ep][bs]#[batch_size,state_dim]
                    central_mixer_outputs_k=self.central_mixer(selected_controller_outputs_k,state_t)
                    mixer_outputs.append(central_mixer_outputs_k.tolist())# arrary of length
                mixer_outputs=np.array(mixer_outputs)
                index_max_mixer_outputs=mixer_outputs.argmax()
                optimal_chosen_actions=chosen_actions[index_max_mixer_outputs]

                return  optimal_chosen_actions


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.learner.train(batch, t_env, episode_num)

    def cuda(self):
        self.controller.cuda()
        self.target_control.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.central_controller is not None:
            self.central_controller.cuda()
            self.target_central_controller.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

    # TODO: Model saving/loading is out of date!
    def save_models(self, path):
        self.controller.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        #th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    def load_models(self, path):
        self.controller.load_models(path)
        # Not quite right but I don't want to save target networks
        #self.target_central_controller.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        #self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
    def print_q_table(self,episode_batch):
        self.learner.print_q_table(episode_batch)











