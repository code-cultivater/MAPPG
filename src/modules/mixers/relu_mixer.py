# From https://github.com/wjh720/QPLEX/, added here for convenience.
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dmaq_si_weight import DMAQ_SI_Weight
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_no_abs import  QMixerNoAbs


class Relu_Mixer(nn.Module):
    def __init__(self, args):
        super(Relu_Mixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim +  self.n_agents

        self.qmix_no_abs=QMixerNoAbs(args)


    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        state_action = th.cat([states, agent_qs], dim=1)

        vdn_out=th.sum(agent_qs,dim=-1,keepdim=True)
        vdn_out=vdn_out.view(bs,-1,1)

        qmix_no_abs_out=self.qmix_no_abs(agent_qs,states)
        qmix_no_abs_out=qmix_no_abs_out.view(bs,-1,1)

        q=qmix_no_abs_out+vdn_out
        return q
class Relu_MixerV3(nn.Module):
    def __init__(self, args):
        super(Relu_Mixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim +  self.n_agents

        self.qmix=QMixer(args)

        hypernet_embed = self.args.central_mixing_embed_dim
        # self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
        #                                    nn.ReLU(),
        #                                    nn.Linear(hypernet_embed, self.n_agents))

        self.V = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, 1))



        self.action_extractors = nn.ModuleList()
        self.num_kernel = args.num_kernel

        for i in range(self.num_kernel):  # multi-head attention
            self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(hypernet_embed, hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(hypernet_embed, hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(hypernet_embed, 1)))  # action

    def hyper_w_final_for_action(self,state_action):
        lambda_weights=[]
        for i in range(self.num_kernel):
            weight=self.action_extractors[i](state_action)
            lambda_weights.append(weight)
        lambdas = th.stack(lambda_weights, dim=1)
        lambdas = lambdas.reshape(-1, self.num_kernel, 1).sum(dim=1)
        lambdas.view(-1, 1)
        return lambdas

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        state_action = th.cat([states, agent_qs], dim=1)

        v = self.V(states)
        v = v.view(bs,-1, 1)

        qmix_out=self.qmix(agent_qs,states)
        qmix_out=qmix_out.view(bs,-1,1)

        # w_final = self.hyper_w_final(states)
        # w_final=w_final.view(-1, self.n_agents)
        # w_final=th.abs(w_final)
        # state_weighted_qs=w_final * agent_qs

        advs=self.hyper_w_final_for_action(state_action)
        advs=advs.view(bs,-1,1)

        # q=v+state_weighted_qs+action_weighted_qs
        # q=th.sum(q,dim=1)
        # q = q.view(bs, -1, 1)

        q=v+qmix_out+advs
        return q


class Relu_MixerV2(nn.Module):
    def __init__(self, args):
        super(Relu_Mixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim +  self.n_agents

        self.qmix=QMixer(args)

        hypernet_embed = self.args.central_mixing_embed_dim
        # self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
        #                                    nn.ReLU(),
        #                                    nn.Linear(hypernet_embed, self.n_agents))

        self.V = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, 1))



        self.action_extractors = nn.ModuleList()
        self.num_kernel = args.num_kernel

        for i in range(self.num_kernel):  # multi-head attention
            self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(hypernet_embed, 1)))  # action

    def hyper_w_final_for_action(self,state_action):
        lambda_weights=[]
        for i in range(self.num_kernel):
            weight=self.action_extractors[i](state_action)
            lambda_weights.append(weight)
        lambdas = th.stack(lambda_weights, dim=1)
        lambdas = lambdas.reshape(-1, self.num_kernel, 1).sum(dim=1)
        lambdas.view(-1, 1)
        return lambdas

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        state_action = th.cat([states, agent_qs], dim=1)

        v = self.V(states)
        v = v.view(bs,-1, 1)

        qmix_out=self.qmix(agent_qs,states)
        qmix_out=qmix_out.view(bs,-1,1)

        # w_final = self.hyper_w_final(states)
        # w_final=w_final.view(-1, self.n_agents)
        # w_final=th.abs(w_final)
        # state_weighted_qs=w_final * agent_qs

        advs=self.hyper_w_final_for_action(state_action)
        advs=advs.view(bs,-1,1)

        # q=v+state_weighted_qs+action_weighted_qs
        # q=th.sum(q,dim=1)
        # q = q.view(bs, -1, 1)

        q=v+qmix_out+advs
        return q



class Relu_MixerV1(nn.Module):
    def __init__(self, args):
        super(Relu_MixerV1, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim

        self.qmix=QMixer(args)

        hypernet_embed = self.args.central_mixing_embed_dim
        # self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
        #                                    nn.ReLU(),
        #                                    nn.Linear(hypernet_embed, self.n_agents))

        self.V = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, 1))



        self.action_extractors = nn.ModuleList()
        self.num_kernel = args.num_kernel

        for i in range(self.num_kernel):  # multi-head attention
            self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(hypernet_embed, hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(hypernet_embed, hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(hypernet_embed, self.n_agents)))  # action

    def hyper_w_final_for_action(self,state_action):
        lambda_weights=[]
        for i in range(self.num_kernel):
            weight=self.action_extractors[i](state_action)
            lambda_weights.append(weight)
        lambdas = th.stack(lambda_weights, dim=1)
        lambdas = lambdas.reshape(-1, self.num_kernel, self.n_agents).sum(dim=1)
        lambdas.view(-1, self.n_agents)
        return lambdas

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, self.n_agents)
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        state_action = th.cat([states, actions], dim=1)

        v = self.V(states)
        v = v.view(bs,-1, 1)

        qmix_out=self.qmix(agent_qs,states)
        qmix_out=qmix_out.view(bs,-1,1)

        # w_final = self.hyper_w_final(states)
        # w_final=w_final.view(-1, self.n_agents)
        # w_final=th.abs(w_final)
        # state_weighted_qs=w_final * agent_qs

        hyper_w_final=self.hyper_w_final_for_action(state_action)
        action_weighted_qs=hyper_w_final*agent_qs.detach()
        action_weighted_qs=th.sum(action_weighted_qs,dim=-1)
        action_weighted_qs=action_weighted_qs.view(bs,-1,1)

        # q=v+state_weighted_qs+action_weighted_qs
        # q=th.sum(q,dim=1)
        # q = q.view(bs, -1, 1)

        q=v+qmix_out+action_weighted_qs
        return q


class Relu_MixerV0(nn.Module):
    def __init__(self, args):
        super(Relu_MixerV0, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim


        hypernet_embed = self.args.hypernet_embed
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents))

        self.V = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, self.n_agents))



        self.action_extractors = nn.ModuleList()
        self.num_kernel = args.num_kernel
        adv_hypernet_embed = self.args.adv_hypernet_embed
        for i in range(self.num_kernel):  # multi-head attention
            self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(adv_hypernet_embed, self.n_agents)))  # action

    def hyper_w_final_for_action(self,state_action):
        lambda_weights=[]
        for i in range(self.num_kernel):
            weight=self.action_extractors[i](state_action)
            lambda_weights.append(weight)
        lambdas = th.stack(lambda_weights, dim=1)
        lambdas = lambdas.reshape(-1, self.num_kernel, self.n_agents).sum(dim=1)
        lambdas.view(-1, self.n_agents)
        return lambdas

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, self.n_agents)
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        state_action = th.cat([states, actions], dim=1)

        v = self.V(states)
        v = v.view(-1, self.n_agents)

        w_final = self.hyper_w_final(states)
        w_final=w_final.view(-1, self.n_agents)
        w_final=th.abs(w_final)
        state_weighted_qs=w_final * agent_qs

        hyper_w_final=self.hyper_w_final_for_action(state_action)
        action_weighted_qs=hyper_w_final*agent_qs.detach()

        q=v+state_weighted_qs+action_weighted_qs
        q=th.sum(q,dim=1)
        q = q.view(bs, -1, 1)
        return q
