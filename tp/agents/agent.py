# SRC
import src.graphicsUtils as graphicsUtils

# RL TP
from tp.utils import Action, State
import gym

# PYTHON
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from random import random
from numpy.random import choice 
from math import exp, sqrt, log
import sys


class Agent(ABC):
    """
    Base class for all of our model-based agents. 
    An agent is an object that can interact with an environment and learn from it.
    """
    def __init__(self, env : gym.Env, **kwargs):
        self.env = env

    @abstractmethod
    def act(self, state, training = True) -> Action:
        """Return the action to take in the given state"""
        pass

    @abstractmethod
    def observe(self, state, action, reward, next_state, done):
        """Observe the transition and possibly stock in memory"""
        pass

    @abstractmethod
    def learn(self):
        """Learn using the memory"""
        pass    

    ### HELPERS ###
    
    def get_possible_actions(self, state : State) -> List[Action]:
        return self.env.gridworld.getPossibleActions(state)


class ValueBasedAgent(Agent):
    """
    Value Based Agent are, in particular, agents that learns using a Q values (and eventually a V value).
    The policy is directly extracted from the Q value (using eps-greedy policy for example).
    Any value based agent class must umplement the getQValue function. This is for the visualization of the Q values.
    """

    @abstractmethod
    def getQValue(self, state: State, action: Action) -> float:
        """Return the Q value of the given state and action"""
        pass



class PolicyBasedAgent(Agent):
    """NOT IMPLEMENTED"""



# You can use those utils functions that represents the policies

def greedy_policy(QValues, state, actions):
    best_action = None
    best_value = -float("inf")
    for action in actions:
        value = QValues[state][action]
        if value > best_value:
            best_action = action
            best_value = value
    if best_action is None:
        raise Exception("No action found for state", state)
    return best_action

def epsilon_greedy_policy(QValues, state, actions, epsilon):
    if random() < epsilon:
        action = choice(actions)
    else:
        action = greedy_policy(QValues, state, actions)
    return action

def boltzmann_policy(QValues, state, actions, temperature = 1):
    values = [QValues[state][action] for action in actions]
    exp_values = [exp(value / temperature) for value in values]
    sum_exp_values = sum(exp_values)
    probs = [exp_value / sum_exp_values for exp_value in exp_values]
    action = choice(actions, p=probs)
    return action

def upper_confidence_bound_policy(QValues : dict, state : State, actions : List[Action], NVisits : dict, N_timesteps : int, c : float = 2):
    """Find the best action according to the UCB policy, which is a tradeoff between exploration and exploitation.

    Args:
        QValues (dict): a dict of dict of Q values
        state (State): the state your in
        actions (List[Action]): the possible actions
        NVisits (dict): a dict of dict of number of visits of each state-action pair
        N_timesteps (int): the number of timesteps since the beginning of training
        
    Returns:
        best_action (Action): the best action according to the UCB policy
    """
    best_action = None
    best_value = -float("inf")
    for action in actions:
        value = QValues[state][action]                                  # exploitation term
        value += c * sqrt(log(NVisits[state][action]) / N_timesteps)    # exploration term
        if value > best_value:
            best_action = action
            best_value = value
    if best_action is None:
        raise Exception("No action found for state", state)
    return best_action

def random_policy(actions):
    return choice(actions)

def manual_policy(actions):
    """
    Get an action from the user (rather than the agent).

    Used for debugging and lecture demos.
    """
    
    action = None
    while True:
        keys = graphicsUtils.wait_for_keys()
        if 'Up' in keys: action = 'north'
        if 'Down' in keys: action = 'south'
        if 'Left' in keys: action = 'west'
        if 'Right' in keys: action = 'east'
        if 'q' in keys: sys.exit(0)
        if action == None: continue
        break
    if action not in actions:
        action = actions[0]
    return action


policies = {
    "greedy" : greedy_policy,
    "epsilon_greedy" : epsilon_greedy_policy,
    "boltzmann" : boltzmann_policy,
    # "upper_confidence_bound" : upper_confidence_bound_policy,
    "random" : random_policy,
    "manual" : manual_policy,
}
