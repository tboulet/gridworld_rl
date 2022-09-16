import random
import src.mdp as mdp
random_agent_factory = 1


class RandomAgent:
    def __init__(self, mdp):
        self.mdp = mdp

    def getAction(self, state):
        return random.choice(self.mdp.getPossibleActions(state))
    def getValue(self, state):
        return 0.0
    def getQValue(self, state, action):
        return 0.0
    def getPolicy(self, state):
        "NOTE: 'random' is a special policy value; don't use it in your code."
        return 'random'
    def update(self, state, action, nextState, reward):
        pass

def random_agent_factory(mdp, opts):
    return RandomAgent(mdp)