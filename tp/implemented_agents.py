from tp.agents.solutions_agents.random import RandomAgentSolution
from tp.agents.solutions_agents.sarsa import SarsaAgentSolution
from tp.agents.solutions_agents.qlearning import QLearningAgentSolution
from tp.agents.solutions_agents.montecarlo import MonteCarloAgentSolution

agents_map = {
    "_random": RandomAgentSolution,
    "_sarsa" : SarsaAgentSolution,
    "_qlearning" : QLearningAgentSolution,
    "_mc" : MonteCarloAgentSolution

}