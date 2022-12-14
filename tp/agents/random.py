from tp.agent import Agent
from random import choice



class RandomAgent(Agent):
    def act(self, state, training = None):
        return choice(self.get_possible_actions(state))

    def observe(self, state, action, reward, next_state, done):
        pass

    def learn(self):
        pass