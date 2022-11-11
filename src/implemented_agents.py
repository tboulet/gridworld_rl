from src.agents.qlearningAgents import qlearning_agent_factory
from src.agents.randomAgents import random_agent_factory



agents_factory_map = {
    "qlearning" : qlearning_agent_factory,
    "random" : random_agent_factory,
}