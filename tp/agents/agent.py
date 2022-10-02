from tp.utils import Action, State

from abc import ABC, abstractmethod
import gym



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
        """Observe the transition and stock in memory"""
        pass

    @abstractmethod
    def learn(self):
        """Learn using the memory"""
        pass    



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