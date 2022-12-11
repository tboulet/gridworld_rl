from tp.agent import (
    Agent, ValueBasedAgent, policies, 
    greedy_policy, 
    epsilon_greedy_policy, 
    random_policy, 
    boltzmann_policy, 
    manual_policy
    )



class QLearningAgent(ValueBasedAgent):
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
        elif self.behaviour_policy == "manual":
            return manual_policy(actions)
        else:
            raise Exception(f"Policy {self.behaviour_policy} not implemented.")
    
    
    def observe(self, state, action, reward, next_state, done):
        self.observe_state(state)
        self.observe_state(next_state)
        # Save observed transition
        self.last_transition = (state, action, reward, next_state, done)


    def learn(self):
        
        if self.last_transition is None:
            raise Exception("No transition to learn from. You called learn() before observe().")
        
        state, action, reward, next_state, done = self.last_transition
        
        if not done:
            target = reward + self.gamma * max(self.QValues[next_state].values())
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
        