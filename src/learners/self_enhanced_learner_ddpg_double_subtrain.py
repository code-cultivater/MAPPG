import copy
import gc

from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
import torch as th
from torch.optim import RMSprop,Adam
import torch.nn.functional as F
from collections import deque
from controllers import REGISTRY as mac_REGISTRY
from torch.distributions import Categorical
from components.epsilon_schedules import DecayThenFlatSchedule
class SelfEnhancedDDPGQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        self.last_target_update_episode = 0

        # self.mixer = None
        # assert args.mixer is not None
        # if args.mixer is not None:
        #     if args.mixer == "vdn":
        #         self.mixer = VDNMixer()
        #     elif args.mixer == "qmix":
        #         self.mixer = QMixer(args)
        #     else:
        #         raise ValueError("Mixer {} not recognised.".format(args.mixer))
        #     self.mixer_params = list(self.mixer.parameters())
        #     self.params += list(self.mixer.parameters())
        #     self.target_mixer = copy.deepcopy(self.mixer)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # Central Q
        # TODO: Clean this mess up!
        self.central_mac = None
        assert self.args.central_mixer == "ff"
        self.central_mixer = QMixerCentralFF(args)
        self.central_mixer_prime = QMixerCentralFF(args)
        assert args.central_mac == "basic_central_mac"
        self.central_mac = mac_REGISTRY[args.central_mac](scheme, args) # Groups aren't used in the CentralBasicController. Little hacky
        self.central_mac_prime = mac_REGISTRY[args.central_mac](scheme, args) # Groups aren't used in the CentralBasicController. Little hacky
        self.target_central_mac = copy.deepcopy(self.central_mac)
        self.target_central_mac_prime= copy.deepcopy(self.central_mac_prime)

        self.critic_params= list(self.central_mac.parameters())
        self.critic_params+=list(self.central_mac_prime.parameters())
        self.critic_params += list(self.central_mixer.parameters())
        self.critic_params +=  list(self.central_mixer_prime.parameters())

        self.params += self.critic_params

        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        self.target_central_mixer_prime = copy.deepcopy(self.central_mixer_prime)

        #self.actor_optimiser= Adam(params=  self.mac_params , lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        #self.critic_optimiser= Adam(params=  self.critic_params , lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.actor_optimiser= Adam(params=  self.mac_params , lr=args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)


        #self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.schedule = DecayThenFlatSchedule(.9, .1, args.epsilon_anneal_time,
                                              decay="linear")
        self.adv_max_rho = 0
        self.critic_loss_rho = 0

        self.num_actor_updates = 1
        self.num_critic_updates = 1

    def train_actor(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [bs,ts,n_agents,n_actions]

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time

        # Max over target Q-Values
        if self.args.double_q:
            raise Exception("No double q for DDPG")
        else:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            _, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)  # [bs,ts,n_agents,1]

            target_mac_out_detach = target_mac_out.clone().detach()
            target_mac_out_detach[avail_actions == 0] = -9999999
            _, target_cur_max_actions = target_mac_out_detach[:, :].max(dim=3, keepdim=True)  # [bs,ts,n_agents,1]





        central_target_mac_out = []
        central_target_mac_out_prime = []
        self.target_central_mac.init_hidden(batch.batch_size)
        self.target_central_mac_prime.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            target_agent_outs_prime = self.target_central_mac_prime.forward(batch, t=t)

            central_target_mac_out.append(target_agent_outs)
            central_target_mac_out_prime.append(target_agent_outs_prime)

        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # [bs,ts,n_agents,n_actions,1]
        central_target_mac_out_prime = th.stack(central_target_mac_out_prime[:], dim=1)  # [bs,ts,n_agents,n_actions,1]

        # Use the Qmix max actions
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:, :], 3,
                                                   cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                             self.args.central_action_embed)).squeeze(
            3)  # [bs,ts,n_agents,1]
        central_target_max_agent_qvals_prime = th.gather(central_target_mac_out_prime[:, :], 3,
                                                         cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                                   self.args.central_action_embed)).squeeze(
            3)  # [bs,ts,n_agents,1]
        # central_target_max_agent_qvals = th.gather(central_target_mac_out[:, :], 3,
        #                                            target_cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
        #                                                                                      self.args.central_action_embed)).squeeze(
        #     3)  # [bs,ts,n_agents,1]
        # central_target_max_agent_qvals_prime = th.gather(central_target_mac_out_prime[:, :], 3,
        #                                                  target_cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
        #                                                                                            self.args.central_action_embed)).squeeze(
        #     3)  # [bs,ts,n_agents,1]



        # Mix
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals,
                                                     batch["state"])  # [bs,ts,1]  batch["state"]:[32,30,48]
        target_max_qvals_prime = self.target_central_mixer_prime(central_target_max_agent_qvals_prime,
                                                                 batch["state"])  # [bs,ts,1]  batch["state"]:[32,30,48]

        baseline = target_max_qvals.repeat(1, 1, self.args.n_agents)[:, :-1].reshape(-1)  # [bs*ts-1*n_agents]
        baseline_prime = target_max_qvals_prime.repeat(1, 1, self.args.n_agents)[:, :-1].reshape(
            -1)  # [bs*ts-1*n_agents]


        # my added 107
        # Use the Qmix max actions
        # target_central_target_max_agent_qvals = th.gather(central_target_mac_out[:, :], 3,
        #                                                   target_cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
        #                                                                                                    self.args.central_action_embed)).squeeze(
        #     3)  # [bs,ts,n_agents,1]
        # target_central_target_max_agent_qvals_prime = th.gather(central_target_mac_out_prime[:, :], 3,
        #                                                         target_cur_max_actions[:, :].unsqueeze(4).repeat(1, 1,
        #                                                                                                          1, 1,
        #                                                                                                          self.args.central_action_embed)).squeeze(
        #     3)  # [bs,ts,n_agents,1]
        #
        # target_max_qvals_target = self.target_central_mixer(target_central_target_max_agent_qvals,
        #                                              batch["state"])  # [bs,ts,1]  batch["state"]:[32,30,48]
        # target_max_qvals_target_prime = self.target_central_mixer_prime(target_central_target_max_agent_qvals_prime,
        #                                                          batch["state"])  # [bs,ts,1]  batch["state"]:[32,30,48]
        #
        # target_baseline = target_max_qvals_target.repeat(1, 1, self.args.n_agents)[:, :-1].reshape(-1)  # [bs*ts-1*n_agents]
        # target_baseline_prime = target_max_qvals_target_prime.repeat(1, 1, self.args.n_agents)[:, :-1].reshape(
        #     -1)  # [bs*ts-1*n_agents]

        # my loss

        mask_ = mask.repeat(1, 1, self.args.n_agents).reshape(-1)  # [bs,ts-1,n_agens]
        qmix_actor_loss = 0
        num_update = 2
        for k in range(num_update):


            mac_out_prob = th.nn.functional.softmax(mac_out, dim=-1)  # [bs,ts,n_agents,n_actions]
            mac_out_prob[avail_actions == 0] == 0.0 + 1e-7
            mac_out_prob = mac_out_prob / mac_out_prob.sum(dim=-1, keepdim=True)
            mac_out_prob[avail_actions == 0] = 0 + 1e-7

            def ss3():
                # sample

                # epsilon_action_num=avail_actions.sum(dim=-1,keepdim=True)
                mac_out_prob_ = mac_out_prob  # *(1-self.mac.action_selector.epsilon)+th.ones_like(mac_out_prob)*self.mac.action_selector.epsilon/epsilon_action_num
                # sample_prob = (1 - 0.95) * mac_out_prob_ + th.ones_like(mac_out_prob_) * (0.05)
                # m = Categorical(mac_out_prob_)
                # agent_actions = m.sample().long().detach().unsqueeze(dim=-1)  # [bs,ts,n_agents,1]
                # sample_prob_taken = th.gather(sample_prob, dim=-1, index=agent_actions)  # [bs,ts,n_agents,1]

                m = Categorical(mac_out_prob_)  # [bs,ts,n_agents,n_actions]
                agent_actions = m.sample().long().detach().unsqueeze(dim=-1)  # [bs,ts,n_agents,1]

                random_numbers = th.rand_like(mac_out_prob_[:, :, :, 0])
                self.epsilon = self.mac.action_selector.epsilon
                pick_random = (random_numbers < 0.05).long().unsqueeze(dim=-1)
                random_actions = Categorical((avail_actions + 1e-9).float()).sample().long().unsqueeze(dim=-1)
                agent_actions = pick_random * random_actions + (1 - pick_random) * agent_actions
                mac_out_prob_taken = th.gather(mac_out_prob, dim=-1, index=agent_actions)  # [bs,ts,n_agents,1]
                central_target_sample_agent_qvals = th.gather(central_target_mac_out[:, :], 3,
                                                              agent_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                                      self.args.central_action_embed)).squeeze(
                    3)  # [bs,ts,n_agents,1]

                central_target_sample_agent_qvals_prime = th.gather(central_target_mac_out_prime[:, :], 3,
                                                                    agent_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                                            self.args.central_action_embed)).squeeze(
                    3)  # [bs,ts,n_agents,1]
                target_sample_qvals = self.target_central_mixer(central_target_sample_agent_qvals,
                                                                batch["state"])  # [bs,ts,1]
                target_sample_qvals_prime = self.target_central_mixer_prime(central_target_sample_agent_qvals_prime,
                                                                            batch["state"])  # [bs,ts,1]
                target_sample_qvals = target_sample_qvals.unsqueeze(dim=-2).repeat(1, 1, self.args.n_agents,
                                                                                   1)  # [bs,ts,n_agents,1]
                target_sample_qvals_prime = target_sample_qvals_prime.unsqueeze(dim=-2).repeat(1, 1, self.args.n_agents,
                                                                                               1)  # [bs,ts,n_agents,1]
                target_sample_qvals = target_sample_qvals[:, :-1].reshape(-1)
                target_sample_qvals_prime = target_sample_qvals_prime[:, :-1].reshape(-1)

                # is_action_eual = (agent_actions == cur_max_actions)
                # is_action_eual=is_action_eual[:,:-1].reshape(-1)
                #is_prob_greater_half = (mac_out_prob_taken > 0.5)
                is_prob_greater_half = (mac_out_prob_taken > 0.8)
                is_prob_greater_half = is_prob_greater_half[:, :-1].reshape(-1)
                mac_out_prob_taken[:, :-1].reshape(-1)[mask_ == 0] = 1.0
                log_mac_out_prob_taken = th.log(mac_out_prob_taken)  # [bs*ts-1*n_agens]
                #log_mac_out_prob_taken = (mac_out_prob_taken) /sample_prob_taken.detach() # [bs*ts-1*n_agens]
                log_mac_out_prob_taken = log_mac_out_prob_taken[:, :-1].reshape(-1)
                # adv=th.where(target_sample_qvals-baseline+0.1>0,30*(target_sample_qvals-baseline+0.1),target_sample_qvals-baseline)
                # adv = th.exp((target_sample_qvals - baseline) * 10)
                min_target_sample_qvals = th.min(target_sample_qvals, target_sample_qvals_prime)
                max_baseline = th.max(baseline, baseline_prime)

                # max_baseline_ = th.max(baseline, baseline_prime)
                # max_baseline_target=th.max(target_baseline,target_baseline_prime)
                # max_baseline=th.max(max_baseline_,max_baseline_target)
                # min_target_sample_qvals = th.mean(  th.stack([target_sample_qvals, target_sample_qvals_prime],dim=-1))
                # max_baseline = th.mean(th.stack([baseline, baseline_prime], dim=-1),dim=-1)

                orignal_adv = min_target_sample_qvals - max_baseline

                # xx=(target_sample_qvals - baseline > 0 )& (~ is_action_eual)
                # for smac
                scale=100#100#*min(0.05/(self.adv_max_rho+0.00001),1)
                adv = th.where((min_target_sample_qvals - max_baseline> 0)& (~ is_prob_greater_half),
                               # th.clamp(th.exp(10 * (target_sample_qvals - baseline)), 0, 10),
                               #th.clamp(th.exp(scale * (min_target_sample_qvals - max_baseline)), 0, 10),
                               th.clamp(th.exp(scale * (min_target_sample_qvals - max_baseline)), 0, 20),

                               #th.clamp(th.pow(10,scale * (min_target_sample_qvals - max_baseline)), 0, 100),
                               th.tensor(0.0, dtype=th.float).to(self.args.device))
                # for matrix
                # scale = 300  # 100#*min(0.05/(self.adv_max_rho+0.00001),1)
                # adv = th.where((min_target_sample_qvals - max_baseline > 0) & (~ is_prob_greater_half),
                #                # th.clamp(th.exp(10 * (target_sample_qvals - baseline)), 0, 10),
                #                # th.clamp(th.exp(scale * (min_target_sample_qvals - max_baseline)), 0, 10),
                #               th.exp(scale * (min_target_sample_qvals - max_baseline)),
                #
                #                # th.clamp(th.pow(10,scale * (min_target_sample_qvals - max_baseline)), 0, 100),
                #                th.tensor(0.0, dtype=th.float).to(self.args.device))


                agent_policy_loss = (log_mac_out_prob_taken * (
                    adv.detach()) * mask_.detach()).sum() / mask_.detach().sum()
                return agent_policy_loss, min_target_sample_qvals, orignal_adv



            ss3_loss, target_sample_qvals, adv = ss3()
            qmix_actor_loss += ss3_loss







        loss = -self.args.qmix_loss * qmix_actor_loss


        # Optimise
        self.actor_optimiser.zero_grad()

        loss.backward()

        grad_norm = th.nn.utils.clip_grad_norm_(self.mac_params, self.args.grad_norm_clip)


        self.actor_optimiser.step()
        self.adv_max_rho= self.adv_max_rho+0.1*( (adv * mask_).max().item()-self.adv_max_rho)



        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("adv_max_rho", self.adv_max_rho, t_env)
            self.logger.log_stat("actor_loss", qmix_actor_loss.item(), t_env)
            self.logger.log_stat("actor_grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("target_sample_max", (target_sample_qvals * mask_).max().item(),
                                 t_env)
            self.logger.log_stat("target_sample_mean",
                                 (target_sample_qvals * mask_).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.logger.log_stat("adv_max", (adv * mask_).max().item(),
                                 t_env)
            self.logger.log_stat("adv_rate", ( ((adv>0)*mask_).sum()/mask_.sum()).item(),
                                 t_env)
            self.logger.log_stat("adv_mean", ((adv * mask_).sum() / mask_.sum()).item(),
                                 t_env)
            self.logger.log_stat("mac_out_prob_max_value_mean",
                                 (th.max(mac_out_prob, dim=-1)[0][:, :-1] * mask).sum().item() / (
                                             mask_elems * self.args.n_agents),
                                 t_env)




    def train_critic(self, batch: EpisodeBatch, t_env: int, episode_num: int):


        rewards = batch["reward"][:, :-1]
        actions = batch["actions"]#[bs,ts,n_agents,1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [bs,ts,n_agents,n_actions]

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions[:,:-1]).squeeze(3)  # Remove the last dim
        chosen_action_qvals = chosen_action_qvals_agents



        # Max over target Q-Values
        if self.args.double_q:
            raise Exception("No double q for DDPG")
        else:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            _, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)#[bs,ts,n_agents,1]


        # Central MAC stuff
        central_mac_out = []
        central_mac_out_prime = []
        self.central_mac.init_hidden(batch.batch_size)
        self.central_mac_prime.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            agent_outs_prime = self.central_mac_prime.forward(batch, t=t)
            central_mac_out.append(agent_outs)
            central_mac_out_prime.append(agent_outs_prime)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        central_mac_out_prime = th.stack(central_mac_out_prime, dim=1)  # Concat over time

        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions[:,:-1].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim
        central_chosen_action_qvals_agents_prime = th.gather(central_mac_out_prime[:, :-1], dim=3, index=actions[:,:-1].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim
        # print(self.central_mac.agent)
        # print(self.target_central_mixer)


        central_target_mac_out = []
        central_target_mac_out_prime = []
        self.target_central_mac.init_hidden(batch.batch_size)
        self.target_central_mac_prime.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            target_agent_outs_prime = self.target_central_mac_prime.forward(batch, t=t)

            central_target_mac_out.append(target_agent_outs)
            central_target_mac_out_prime.append(target_agent_outs_prime)

        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # [bs,ts,n_agents,n_actions,1]
        central_target_mac_out_prime = th.stack(central_target_mac_out_prime[:], dim=1)  # [bs,ts,n_agents,n_actions,1]

        # Use the Qmix max actions
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:,:], 3, cur_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)#[bs,ts,n_agents,1]
        central_target_max_agent_qvals_prime = th.gather(central_target_mac_out_prime[:,:], 3, cur_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)#[bs,ts,n_agents,1]




        # Mix
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals, batch["state"])#[bs,ts,1]  batch["state"]:[32,30,48]
        target_max_qvals_prime = self.target_central_mixer_prime(central_target_max_agent_qvals_prime, batch["state"])#[bs,ts,1]  batch["state"]:[32,30,48]

        # Calculate 1-step Q-Learning targets
        min_target_max_qvals=(th.min(target_max_qvals, target_max_qvals_prime))
        targets = rewards + self.args.gamma * (1 - terminated) * min_target_max_qvals[:, 1:]




        # Training central Q
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1])
        central_chosen_action_qvals_prime = self.central_mixer_prime(central_chosen_action_qvals_agents_prime, batch["state"][:, :-1])

        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_td_error_prime = (central_chosen_action_qvals_prime - targets.detach())

        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_masked_td_error_prime = central_td_error_prime * central_mask

        central_loss = (central_masked_td_error ** 2).sum() / mask.sum()+(central_masked_td_error_prime ** 2).sum() / mask.sum()

        loss = self.args.central_loss * central_loss

        # Optimise
        self.critic_optimiser.zero_grad()
        loss.backward()

        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)


        self.critic_optimiser.step()
        self.critic_loss_rho = self.critic_loss_rho + 0.1 * (loss.item() - self.critic_loss_rho)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss_rho", self.critic_loss_rho, t_env)
            th.set_printoptions(profile="full")
            print("mac", th.nn.functional.softmax(mac_out,dim=-1)[0][0],th.nn.functional.softmax(mac_out,dim=-1)[0][0].max(dim=-1)[1])
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            self.logger.log_stat("target_max", (targets * mask).max().item(),
                                 t_env)



    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.train_critic(batch, t_env, episode_num)
        for _ in range(1):
            self.train_actor(batch,t_env,episode_num)

        #self.train_critic(batch, t_env, episode_num)
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)

        if self.central_mac is not None:
            self.target_central_mac.load_state(self.central_mac)
            self.target_central_mac_prime.load_state(self.central_mac_prime)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())
        self.target_central_mixer_prime.load_state_dict(self.central_mixer_prime.state_dict())

        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

        if self.central_mac is not None:
            self.central_mac.cuda()
            self.target_central_mac.cuda()
            self.central_mac_prime.cuda()
            self.target_central_mac_prime.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()
        self.central_mixer_prime.cuda()
        self.target_central_mixer_prime.cuda()

    def save_models(self, path):
        # self.mac.save_models(path)
        # if self.mixer is not None:
        #     th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        # th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        pass;

    def load_models(self, path):
        # self.mac.load_models(path)
        # # Not quite right but I don't want to save target networks
        # self.target_mac.load_models(path)
        # if self.mixer is not None:
        #     self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        pass;


    def print_q_table(self,episode_batch,t):
        '''
        :param episode_batch: {state：[batch_size,*shape], obs:[batch_size,n_agents,*shape]}
        :return:
        '''

        self.mac.init_hidden(episode_batch.batch_size)
        self.central_mac.init_hidden(episode_batch.batch_size)
        # [batch_size,n_agents,n_actions]
        agent_outs = self.mac.forward(episode_batch, t=0)
        import torch
        agent_outs=torch.nn.functional.softmax(agent_outs, dim=-1)
        central_agent_outs = self.central_mac.forward(episode_batch, t=0).squeeze(dim=-1)
        import numpy as np
        pay_off_matrix = np.zeros([self.args.n_actions, self.args.n_actions])
        pay_off_matrix_for_central = np.zeros([self.args.n_actions, self.args.n_actions])

        if(self.args.n_agents==2):

            for i in range(self.args.n_actions):
                for j in range(self.args.n_actions):
                    actions = np.array([[[i], [j]]])
                    import torch as th
                    actions = th.from_numpy(actions)
                    if (self.args.use_cuda):
                        actions = actions.cuda()
                    actions = actions.long()
                    chosen_action_ind_qvals = th.gather(agent_outs, dim=2, index=actions).squeeze(2)
                    #chosen_action_qvals = self.mixer(chosen_action_ind_qvals, episode_batch["state"][:, :-1])

                    chosen_central_action_ind_qvals = th.gather(central_agent_outs, dim=2, index=actions).squeeze(2)
                    chosen_acttion_qvals_for_lica = self.central_mixer(chosen_central_action_ind_qvals,
                                                                       episode_batch["state"][:, :-1])

                    #pay_off_matrix[i][j] = chosen_action_qvals.cpu().detach().numpy().squeeze().tolist()
                    pay_off_matrix_for_central[i][
                        j] = chosen_acttion_qvals_for_lica.cpu().detach().numpy().squeeze().tolist()


        elif(self.args.n_agents==3):
            pay_off_matrix_for_central = np.zeros([3, 3])
            action_list=np.array([
                [[[0], [0],[0]]],
                [[[0], [1], [1]]],
                [[[0], [2], [1]]],
                [[[1], [0], [1]]],
                [[[1], [1], [1]]],
                [[[1], [2], [1]]],
                [[[2], [0], [1]]],
                [[[2], [1], [1]]],
                [[[2], [2], [1]]]
                                  ])
            for k in range(9):
                actions = action_list[k]
                import torch as th
                actions = th.from_numpy(actions)
                if (self.args.use_cuda):
                    actions = actions.cuda()
                actions = actions.long()
                chosen_action_ind_qvals = th.gather(agent_outs, dim=2, index=actions).squeeze(2)
                # chosen_action_qvals = self.mixer(chosen_action_ind_qvals, episode_batch["state"][:, :-1])

                chosen_central_action_ind_qvals = th.gather(central_agent_outs, dim=2, index=actions).squeeze(2)
                chosen_acttion_qvals_for_lica = self.central_mixer(chosen_central_action_ind_qvals,
                                                                   episode_batch["state"][:, :-1])

                # pay_off_matrix[i][j] = chosen_action_qvals.cpu().detach().numpy().squeeze().tolist()
                pay_off_matrix_for_central[int(np.floor(k/3))][k%3] = chosen_acttion_qvals_for_lica.cpu().detach().numpy().squeeze().tolist()
        elif (self.args.n_agents > 3):
            pay_off_matrix_for_central = np.zeros([3, 3])
            # action_list = np.array([
            #     [[[0], [0], [0]]],
            #     [[[0], [1], [1]]],
            #     [[[0], [2], [1]]],
            #     [[[1], [0], [1]]],
            #     [[[1], [1], [1]]],
            #     [[[1], [2], [1]]],
            #     [[[2], [0], [1]]],
            #     [[[2], [1], [1]]],
            #     [[[2], [2], [1]]]
            # ])#(9, 1, 3, 1)
            action_list=np.ones([9,1,self.args.n_agents,1])
            action_list[0][0][:][0]=0
            action_list[0][0][0][0],action_list[0][0][1][0]=0,0
            action_list[1][0][0][0], action_list[1][0][1][0] = 0, 1
            action_list[2][0][0][0], action_list[2][0][1][0] = 0, 2
            action_list[3][0][0][0], action_list[3][0][1][0] = 1, 0
            action_list[4][0][0][0], action_list[4][0][1][0] = 1, 1
            action_list[5][0][0][0], action_list[5][0][1][0] = 1, 2
            action_list[6][0][0][0], action_list[6][0][1][0] = 2, 0
            action_list[7][0][0][0], action_list[7][0][1][0] = 2, 1
            action_list[8][0][0][0], action_list[8][0][1][0] = 2, 2

            for k in range(9):
                actions = action_list[k]
                import torch as th
                actions = th.from_numpy(actions)
                if (self.args.use_cuda):
                    actions = actions.cuda()
                actions = actions.long()
                chosen_action_ind_qvals = th.gather(agent_outs, dim=2, index=actions).squeeze(2)
                # chosen_action_qvals = self.mixer(chosen_action_ind_qvals, episode_batch["state"][:, :-1])

                chosen_central_action_ind_qvals = th.gather(central_agent_outs, dim=2, index=actions).squeeze(2)
                chosen_acttion_qvals_for_lica = self.central_mixer(chosen_central_action_ind_qvals,
                                                                   episode_batch["state"][:, :-1])

                # pay_off_matrix[i][j] = chosen_action_qvals.cpu().detach().numpy().squeeze().tolist()
                pay_off_matrix_for_central[int(np.floor(k / 3))][
                    k % 3] = chosen_acttion_qvals_for_lica.cpu().detach().numpy().squeeze().tolist()


        print("--------------------------------")
        print(agent_outs)
        print(torch.max(agent_outs,dim=-1)[1])
        # print(pay_off_matrix)
        print("pay_off_matrix_for_central\n",pay_off_matrix_for_central)