from tp.agent import Agent, ValueBasedAgent, policies, greedy_policy, epsilon_greedy_policy, random_policy, boltzmann_policy
from random import choice, random



class SarsaAgent(ValueBasedAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Hyperparameters
        self.gamma = 0.9
        self.epsilon = 0.1
        self.learning_rate = 0.1
        self.behaviour_policy = "boltzmann"
        assert self.behaviour_policy in policies.keys(), f"Policy {self.behaviour_policy} not implemented : not in {policies.keys()}."
        # Other
        self.QValues = {}
        self.last_transition : tuple = None
        self.transition : tuple = None

        
    def act(self, state, training = None):
        self.observe_state(state)
        actions = self.get_possible_actions(state)
        if self.behaviour_policy == "epsilon_greedy":
            return epsilon_greedy_policy(self.QValues, state, actions, self.epsilon)
        elif self.behaviour_policy == "greedy":
            return greedy_policy(self.QValues, state, actions)
        elif self.behaviour_policy == "random":
            return random_policy(actions)
        elif self.behaviour_policy == "boltzmann":
            return boltzmann_policy(self.QValues, state, actions, temperature=0.1)
        else:
            raise Exception(f"Policy {self.behaviour_policy} not implemented for SARSA.")
    
    
    def observe(self, state, action, reward, next_state, done):
        self.observe_state(state)
        self.observe_state(next_state)
        # Save transition
        self.last_transition = self.transition
        self.transition = (state, action, reward, next_state, done)


    def learn(self):
        
        # Pass learning for first transition.
        if self.transition is None or self.last_transition is None:
            return
        
        state, action, reward, next_state, done = self.last_transition
        next_state, next_action, next_reward, next_next_state, next_done = self.transition
        
        if not done:
            # self.observe_state(next_state)
            # next_action = self.act(next_state) # We compute the next action from the current policy. We could also implement this by waiting for next action to be observed but a bit more complicated to implement.
            next_action = next_action
            target = reward + self.gamma * self.QValues[next_state][next_action]
        else:
            target = reward
        
        self.QValues[state][action] += self.learning_rate * (target - self.QValues[state][action])
        
        
    ### getQValues for vizualisation
    def getQValue(self, state, action):
        try:
            return self.QValues[state][action]
        except KeyError:
            return 0
        
        
    ### Helpers ###
    def observe_state(self, state):
        """If state was never seen, save its Q values as 0.

        Args:
            state (State): the state to observe
        """
        if state not in self.QValues:
            self.QValues[state] = {action : 0 for action in self.get_possible_actions(state)}
        