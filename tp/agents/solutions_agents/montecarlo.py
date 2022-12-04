from tp.agent import Agent, ValueBasedAgent, policies, greedy_policy, epsilon_greedy_policy, random_policy, boltzmann_policy
from random import choice, random



class MonteCarloAgentSolution(ValueBasedAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Hyperparameters
        self.gamma = 0.9
        self.epsilon = 0.2
        self.learning_rate = 0.1
        self.behaviour_policy = "epsilon_greedy"
        assert self.behaviour_policy in policies.keys(), f"Policy {self.behaviour_policy} not implemented : not in {policies.keys()}."
        # Other
        self.QValues = {}
        self.timesteps_to_states_actions = {} # mapping from timestep to tuple of (state, action)
        self.timesteps_to_rewards = {} # mapping from timestep to reward
        self.timestep = 0
        self.is_done = False
        
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
            return boltzmann_policy(self.QValues, state, actions, temperature=1)
        else:
            raise Exception(f"Policy {self.behaviour_policy} not implemented.")
    
    
    def observe(self, state, action, reward, next_state, done):
        self.is_done = done
        self.observe_state(state)
        self.observe_state(next_state)
        self.timesteps_to_rewards[self.timestep] = reward
        self.timesteps_to_states_actions[self.timestep] = (state, action)


    def learn(self):
        if self.is_done:
            future_reward = 0
            for t in range(self.timestep, -1, -1):
                state, action = self.timesteps_to_states_actions[t]
                reward = self.timesteps_to_rewards[t]
                future_reward = reward + self.gamma * future_reward
                self.QValues[state][action] += self.learning_rate * (future_reward - self.QValues[state][action])
            self.reset_episode()
        else:
            self.timestep += 1
            
        
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
    
    def reset_episode(self):
        self.timesteps_to_states_actions = {}
        self.timesteps_to_rewards = {}
        self.timestep = 0
        self.is_done = False