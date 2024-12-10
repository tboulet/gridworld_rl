from math import exp
from tp.agent import Agent, ValueBasedAgent, policies, greedy_policy, epsilon_greedy_policy, random_policy, boltzmann_policy
from random import choice, random
import numpy as np


class REINFORCEAgent(ValueBasedAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Hyperparameters
        self.gamma = 0.9
        self.temperature = 1
        self.learning_rate = 0.1
        # Other
        self.weights = {}
        self.timesteps_to_states_actions = {} # mapping from timestep to tuple of (state, action)
        self.timesteps_to_rewards = {} # mapping from timestep to reward
        self.timestep = 0
        self.is_done = False
        
    def act(self, state, training = None):
        self.observe_state(state)
        pi_s = self.get_pi_s(self.weights, state)
        action = np.random.choice(list(pi_s.keys()), p=list(pi_s.values()))
        return action
    
    def observe(self, state, action, reward, next_state, done):
        self.is_done = done
        self.observe_state(state)
        self.observe_state(next_state)
        self.timesteps_to_rewards[self.timestep] = reward
        self.timesteps_to_states_actions[self.timestep] = (state, action)


    def learn(self):
        if self.is_done:
            def b(state):
                return 0  # TODO : implement baseline
            future_reward = 0
            for t in range(self.timestep, -1, -1):
                state, action = self.timesteps_to_states_actions[t]
                pi_s = self.get_pi_s(self.weights, state)
                reward = self.timesteps_to_rewards[t]
                future_reward = reward + self.gamma * future_reward
                future_reward_centered = future_reward - b(state)
                for a in self.get_possible_actions(state):
                    delta_a_action = 1 if a == action else 0
                    self.weights[state][a] = self.weights[state][a] + self.learning_rate * future_reward_centered * (delta_a_action - pi_s[a])
            self.reset_episode()
        else:
            self.timestep += 1
            
        
    ### getQValues for vizualisation
    def getQValue(self, state, action):
        try:
            return self.get_pi_s(self.weights, state)[action]
            return self.weights[state][action]
        except KeyError:
            return 0
        
        
    ### Helpers ###
    def observe_state(self, state):
        """If state was never seen, save its Q values as 0.

        Args:
            state (State): the state to observe
        """
        INITIALIZATION_WEIGHTS_METHOD = "zeros"
        if state not in self.weights:
            if INITIALIZATION_WEIGHTS_METHOD == "zeros":
                self.weights[state] = {action : 0 for action in self.get_possible_actions(state)}
            # TODO : implement other initialization methods
            elif INITIALIZATION_WEIGHTS_METHOD == "random":
                raise NotImplementedError("Random initialization of weights not implemented")
            else:
                raise Exception(f"Initialization method {INITIALIZATION_WEIGHTS_METHOD} not implemented.")
    
    def get_pi_s(self, weights, state):
        pi_s = {}
        sum_weights = sum([exp(weights[state][a]) for a in self.get_possible_actions(state)])
        for action in self.get_possible_actions(state):
            pi_s[action] = exp(weights[state][action]) / sum_weights
        return pi_s

    def reset_episode(self):
        self.timesteps_to_states_actions = {}
        self.timesteps_to_rewards = {}
        self.timestep = 0
        self.is_done = False