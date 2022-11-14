from tp.agents.random import RandomAgent
from tp.agents.sarsa import SarsaAgent
from tp.agents.qlearning import QLearningAgent
from tp.agents.montecarlo import MonteCarloAgent

agents_map = {
    "random": RandomAgent,
    "sarsa" : SarsaAgent,
    "qlearning" : QLearningAgent,
    "mc" : MonteCarloAgent
    
}