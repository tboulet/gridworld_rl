from tp.agent import Agent, ValueBasedAgent, policies, greedy_policy, epsilon_greedy_policy, random_policy, boltzmann_policy
from random import choice, random



class MonteCarloAgent(ValueBasedAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Hyperparameters
        self.gamma = 0.9
        self.epsilon = 0.2
        self.learning_rate = 0.1
        # Q Values are stored as a dictionary of dictionaries
        self.QValues = {}
        
    def act(self, state, training = None) :
        return your_action
    
    def observe(self, state, action, reward, next_state, done):
        pass

    def learn(self):
        pass            
        
    ### getQValues for vizualisation
    def getQValue(self, state, action) -> float:
        return your_QValue
        
        
    ### Helpers ###
    def observe_state(self, state):
        """If state was never seen, save its Q values as 0.

        Args:
            state (State): the state to observe
        """
        if state not in self.QValues:
            self.QValues[state] = {action : 0 for action in self.get_possible_actions(state)}