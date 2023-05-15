from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
from gym import spaces


class QuadraticGame(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)
        self.n_agents = 2


        self.action_space= [spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32) for _ in range(self.n_agents)]
        # Define the agents and actions

        self.n_actions = 1
        self.episode_limit = 1

        def max_two_quadratic(act_1,act_2):
            quard_1=0.8*(-((act_1+5)/3)**2-((act_2+5)/3)**2)
            quard_2=1.0*(-((act_1-5)/1)**2-((act_2-5)/1)**2)+10
            return  max(quard_1,quard_2)



        self.func = max_two_quadratic

        self.state = np.ones(5)

    def reset(self):
        """ Returns initial observations and states"""
        return self.state, self.state

    def step(self, actions):
        """ Returns reward, terminated, info """
        #reward = self.payoff_matrix[actions[0], actions[1]]
        reward = self.func(actions[0], actions[1])


        info = {}
        terminated = True
        info["episode_limit"] = False

        return reward, terminated, info

    def get_obs(self):
        return [self.state for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.get_state_size()

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.state)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def get_stats(self):
        raise NotImplementedError

    def get_env_info(self):

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces":self.action_space
                    }

        return env_info

