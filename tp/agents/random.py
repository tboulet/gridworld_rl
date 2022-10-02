from tp.agents.agent import Agent
from random import choice



class RandomAgent(Agent):
    def act(self, state, training = None):
        return choice(self.env.gridworld.getPossibleActions(state))

    def observe(self, state, action, reward, next_state, done):
        pass

    def learn(self):
        pass