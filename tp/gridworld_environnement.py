# gridworld_environnement.py
# ------------
# Attribution Information: The code used is partially extracted and adapted from the Pacman AI projects.
# The Pacman AI projects were developed at UC Berkeley for the CS188 Intro to AI course.
# Link to the original Berkeley course and Pacman Projects: http://ai.berkeley.edu.
# ------------


# ENV
from src.environment import GridworldEnvironment
import src.graphicsGridworldDisplay as graphicsGridworldDisplay
from tp.environnement.helper import *
from grids import getGrid
# AGENT
from tp.agents.agent import Agent, ValueBasedAgent, PolicyBasedAgent
# PYTHON
from tp.utils import State, Action
import sys
from typing import Any, Tuple
import gym


class Env(gym.Env):
    """
    The environment class. It contains the methods reset(), step() and render().
    """

    def __init__(self, env_options):

        # Init env
        self.env_options = env_options
        mdp = getGrid(env_options.grid)
        mdp.setLivingReward(env_options.livingReward)
        mdp.setNoise(env_options.noise)
        self.gridworld = GridworldEnvironment(mdp)

        # Init display
        self.display = graphicsGridworldDisplay.GraphicsGridworldDisplay(mdp, env_options.gridSize, env_options.speed)
        try:
            self.display.start()
        except KeyboardInterrupt:
            sys.exit(0)
    

    def reset(self) -> State:
        self.gridworld.reset()
        self.state = self.gridworld.getCurrentState()
        return self.state


    def step(self, action: Action) -> Tuple[State, float, bool, Any]:
        state = self.gridworld.getCurrentState()
        if action not in self.gridworld.getPossibleActions(state):
            raise Exception(f"Action {action} not possible in state {state}), possible actions are {self.gridworld.getPossibleActions(state)}")       
        next_state, reward = self.gridworld.doAction(action)
        done = len(self.gridworld.getPossibleActions(next_state)) == 0
        return next_state, reward, done, {}

    
    def render(self, agent : Agent):
        
        env_options = self.env_options
        if not env_options.quiet:
            state = self.gridworld.getCurrentState()
            
            # For value based agents, we display the V or Q values
            if isinstance(agent, ValueBasedAgent): 
                if "getQValue" in dir(agent):
                    self.display.displayQValues(agent, state, "CURRENT Q-VALUES")
                elif "getValue" in dir(agent):
                    self.display.displayValues(agent, state, "CURRENT VALUES")
                else:
                    print("WARNING : agent has no getQValues or getValues method in spite being a ValueBasedAgent")
            
            # For policy based agents, we display the probs
            elif isinstance(agent, PolicyBasedAgent):
                raise NotImplementedError("PolicyBasedAgent not implemented")

            # For other agents (eg random), we display empty values
            else:
                self.display.displayNullValues(state)

