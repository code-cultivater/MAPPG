import numpy as np
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.functional import softmax
from .epsilon_schedules import DecayThenFlatSchedule
import  copy

REGISTRY = {}
class UncertaintyBasedActionSelector():

    def __init__(self,critic_list,args):
        self.critic_list=critic_list

        self.args = args
        self.cnt=0
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        # self.cross_schedule = DecayThenFlatSchedule(args.cross_epsilon_start, args.cross_epsilon_finish, args.cross_epsilon_anneal_time,
        #                                       decay="linear")
        self.epsilon = self.schedule.eval(0)
        #self.cross_epsilon = self.cross_schedule.eval(0)


    def select_action(self, actor_out,ep_batch, avail_actions, t_ep,t_env, test_mode=False):
        '''

        :param actor_out: (ep_batch.batch_size, self.n_agents, self.args.n_actions)
        :param avail_actions:(ep_batch.batch_size, self.n_agents, self.args.n_actions)
        :param t_env:
        :param test_mode:
        :return:picked_actions:(ep_batch.batch_size, self.n_agents)
        '''
        explore_flag=True
        self.cnt+=1
        self.epsilon = self.schedule.eval(t_env)
        #self.cross_epsilon = self.cross_schedule.eval(t_env)
        actor_out=actor_out.clone()
        #actor_out[avail_actions==0]=-9999999
        _,cur_max_actions=th.max(actor_out,dim=-1)# (ep_batch.batch_size, self.n_agents)
        old_cur_max_actions=copy.deepcopy(cur_max_actions)


        #"------"
        cur_max_actions_one_hot = th.nn.functional.one_hot(cur_max_actions,
                                                           self.args.n_actions)  # (ep_batch.batch_size, self.n_agents, self.args.n_actions)
        ep_batch_with_selected_actions = copy.deepcopy(ep_batch)
        # if(self.cnt%200<2 and self.args.name=="pr_q" and t_ep==0):
        #     print("before max replacement: ",ep_batch_with_selected_actions["actions_onehot"][:,t_ep])
        ep_batch_with_selected_actions["actions_onehot"][:, t_ep] = cur_max_actions_one_hot
        # if (self.cnt % 200 < 2 and self.args.name=="pr_q" and t_ep==0):
        #     print("cur_max_actions: ", ep_batch_with_selected_actions["actions_onehot"][:, t_ep])

        critic_out_list = []
        for critic in self.critic_list:
            critic_out = critic(ep_batch_with_selected_actions, t_ep)
            critic_out = critic_out.view(actor_out.size())  # (ep_batch.batch_size, self.n_agents, self.args.n_actions)
            critic_out[avail_actions == 0] = -9999999
            critic_out_list.append(critic_out)
        explore = th.rand(1)
        if(not test_mode):

            if(explore < self.epsilon):

                #pick_random = (random_numbers < self.epsilon).long()  # [batch_size,n_agents]
                random_actions = Categorical(avail_actions.float()).sample().long().to(
                    self.args.device)  # [batch_size,n_agent]

                cur_max_actions =random_actions# pick_random * random_actions + (1 - pick_random) * cur_max_actions  # [batch_size,n_agent]

            else:
                # cur_max_actions_one_hot=th.nn.functional.one_hot(cur_max_actions,self.args.n_actions)#(ep_batch.batch_size, self.n_agents, self.args.n_actions)
                # ep_batch_with_selected_actions=copy.deepcopy(ep_batch)
                # # if(self.cnt%200<2 and self.args.name=="pr_q" and t_ep==0):
                # #     print("before max replacement: ",ep_batch_with_selected_actions["actions_onehot"][:,t_ep])
                # ep_batch_with_selected_actions["actions_onehot"][:,t_ep]=cur_max_actions_one_hot
                # # if (self.cnt % 200 < 2 and self.args.name=="pr_q" and t_ep==0):
                # #     print("cur_max_actions: ", ep_batch_with_selected_actions["actions_onehot"][:, t_ep])
                #
                # critic_out_list=[]
                # for critic in self.critic_list:
                #     critic_out=critic(ep_batch_with_selected_actions,t_ep)
                #     critic_out=critic_out.view(actor_out.size())#(ep_batch.batch_size, self.n_agents, self.args.n_actions)
                #     critic_out[avail_actions==0]=-9999999
                #     critic_out_list.append(critic_out)
                explore_flag = False
                seleted_critic_out_index=th.randint(0,self.args.critic_ensemble_num,(1,))[0]
                current_action_value=critic_out_list[seleted_critic_out_index][0][0][cur_max_actions[0][0]]
                seleted_critic_out=(critic_out_list[seleted_critic_out_index]-current_action_value).detach().clone()# (bs, n_agents, args.n_actions)
                cur_max_actions_one_hot_mask=cur_max_actions_one_hot.detach().clone()
                cur_max_actions_one_hot_mask[:,0,:]=cur_max_actions_one_hot_mask[:,0,:]*0
                seleted_critic_out[cur_max_actions_one_hot_mask==1]=-9999999# (bs, n_agents, args.n_actions)
                seleted_critic_out=seleted_critic_out.view(-1,self.args.n_agents*self.args.n_actions)
                softmax_seleted_critic_out=th.softmax(seleted_critic_out,dim=-1)
                sample_index=Categorical(softmax_seleted_critic_out).sample().long().to(self.args.device)[0]
                action_index=(sample_index)%self.args.n_actions
                agent_index=(sample_index-action_index)/self.args.n_actions
                cur_max_actions[0][agent_index.long()] =action_index.long()   # (ep_batch.batch_size, self.n_agents)




                #explore_for_cross_line = th.rand(1)
                # if (explore_for_cross_line < self.cross_epsilon):
                #     #cur_max_actions  (ep_batch.batch_size, self.n_agents)
                #     explore_agent=th.randint(0,self.args.n_agents,(1,))
                #     random_actions = Categorical(avail_actions[0][explore_agent].float()).sample().long().to(
                #         self.args.device)  # [batch_size,n_agent]
                #     cur_max_actions[0][explore_agent]=random_actions
                #     if (self.cnt % 200 < 2 and self.args.name == "pr_q" and t_ep == 0):
                #         print("explore_for_cross_line")
                # else:





                # max_actions,max_actions_index=seleted_critic_out.max(dim=-1)#(bs, n_agents)
                # max_actions_index=max_actions_index[0]
                # _,max_agent_index=max_actions.max(dim=-1)
                # max_agent_index=max_agent_index[0]
                # cur_max_actions=cur_max_actions#(ep_batch.batch_size, self.n_agents)
                # cur_max_actions[0][max_agent_index]=max_actions_index[max_agent_index]#(ep_batch.batch_size, self.n_agents)
                # if (self.cnt % 200 < 2 and self.args.name == "pr_q" and t_ep == 0):
                #     print("greedy")




                #-------random-----------
                # random_actions = Categorical(avail_actions.float()).sample().long().to(self.args.device)
                # cur_max_actions=random_actions

                #-------multinormial----------
                # critic_out_list = []
                # for critic in self.critic_list:
                #     critic_out = critic(ep_batch_with_selected_actions, t_ep)
                #     critic_out = critic_out.view(
                #         actor_out.size())  # (ep_batch.batch_size, self.n_agents, self.args.n_actions)
                #     critic_out[avail_actions == 0] = -99999999
                #     critic_out_list.append(critic_out)
                # #critic_out_list=th.stack(critic_out_list,dim=-1)#(ep_batch.batch_size, self.n_agents, self.args.n_actions,self.critic_ensemble_num)
                # #max_critic_out_list=critic_out_list.max(dim=-1)[0]#(ep_batch.batch_size, self.n_agents, self.args.n_actions）
                # max_critic_out_list = critic_out_list[th.randint(0,self.args.critic_ensemble_num,(1,))[0]]
                # max_critic_out_list=max_critic_out_list.view(-1,self.args.n_agents*self.args.n_actions)
                # max_critic_out_list=th.softmax(max_critic_out_list/3,dim=-1)
                #
                # sample_index=Categorical(max_critic_out_list).sample().long().to(self.args.device)#(batch_size）
                # sample_index=sample_index[0]
                # action_index=(sample_index)%self.args.n_actions
                # agent_index=(sample_index-action_index)/self.args.n_actions
                # cur_max_actions[0][agent_index.long()] =action_index.long()   # (ep_batch.batch_size, self.n_agents)

            if (self.cnt % 200 < 2 and self.args.name == "pr_q" and t_ep == 0):
                print("t_env:",t_env)
                print("critic_out_list:\n",critic_out_list)
                print("actor_out:\n",actor_out)

        picked_actions=cur_max_actions
        #print("picked_actions: ",picked_actions)
        #if (self.cnt % 200 < 2 and self.args.name == "pr_q" and t_ep == 0):
        #print("cur_max_actions：{}, picked_actions: {}".format(cur_max_actions,picked_actions))
        if (self.cnt % 200 < 2 and self.args.name == "pr_q" and t_ep == 0):
            print("explore, old_cur_max_actions, picked_actions",explore < self.epsilon,old_cur_max_actions, picked_actions)



        return   picked_actions,explore_flag

REGISTRY["uncertainty_action_selector"] = UncertaintyBasedActionSelector








class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        print(masked_policies)
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        # Was there so I used it
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        '''
        :param agent_inputs:[batch_size=1,n_agents,n_actions]
        :param avail_actions:[batch_size=1,n_agents,n_actions]
        :param t_env:
        :param test_mode:
        :return:#[batch_size=1,n_agent]
        '''
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0


        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()# [batch_size,n_agents,n_actions]
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:,:,0])# [batch_size,n_agents]
        pick_random = (random_numbers < self.epsilon).long()# [batch_size,n_agents]
        random_actions = Categorical(avail_actions.float()).sample().long().to(self.args.device)  #   [batch_size,n_agent]

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]#   [batch_size,n_agent]
        return picked_actions#[batch,n_agent]



REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

class E_EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.actor_ensemble_num=args.actor_ensemble_num
        self.epsilon_greedy_action_selector=EpsilonGreedyActionSelector(args)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        '''
        :param agent_inputs:[batch_size=1,n_agents,n_actions]
        :param avail_actions:
        :param t_env:
        :param test_mode:
        :return:#[batch_size=1,n_agent]
        '''
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        picked_actions=[]
        for i in range(self.actor_ensemble_num):
            picked_action=self.epsilon_greedy_action_selector.select_action(agent_inputs[i],avail_actions,t_env,test_mode)
            picked_actions.append(picked_action)
        return picked_actions#[batch,n_agent]


REGISTRY["e_epsilon_greedy"] = E_EpsilonGreedyActionSelector

class PolicyEpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        # Was there so I used it
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_qs, agent_pis, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        # masked_q_values = agent_qs.clone()
        # masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_qs[:,:,0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        # max_action = th.abs(masked_q_values - agent_pis).argmin(dim=2)
        masked_agent_pis = agent_pis.clone()
        masked_agent_pis[avail_actions == 0.0] = -float("inf")
        max_action = masked_agent_pis.argmax(dim=2)
        picked_actions = pick_random * random_actions + (1 - pick_random) * max_action
        return picked_actions

REGISTRY["policy_epsilon_greedy"] = PolicyEpsilonGreedyActionSelector
