from tp.agent import Agent
from random import choice



class RandomAgent(Agent):
        
    def act(self, state, training = None) :
        return your_action
    
    def observe(self, state, action, reward, next_state, done):
        ...

    def learn(self):
        ...            