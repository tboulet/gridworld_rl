# TP Gridworld RL
This project implements a simple yet rich and variable gridworld environment for reinforcement learning, as well as tabular agents such as Q-learning and SARSA. 


# The environnement

The environnement is a gridworld with a start and an end. The agent can move in 4 directions (up, down, left, right) and can only move on the grid. Its objective (usually) is to reach the end state.

The map is defined by the argument --grid and implemented in the grid.py file (default: 'book').

| Square state      | Effect            | Symbol in grid.py |
| -----------       | -----------       | ----------- | 
| Start             | Starting point    |'S' |
| Wall              | Can't go          | '#'|
| Free space        |                   | ' '|
| End bonus         | End episode and give positive reward           |n (with n an integer)  |
| End malus         | End episode and give negative reward              |-n  |


Additional environnement parameters:
 - --livingReward : reward given at each step (default: 0)
 - --noise : probability of moving in a random direction (default: 0.2)


### Observations/States
The environnement is fully observable (one state = one observation). An observation or a state is simply a tuple (x,y) representing the position of the agent on the grid.

### Actions

### Rewards
The reward is a sparse reward given when the agent step on an end state. It can be negative (punishment) or positive (reward).

A dense reward is also possibly given when the livingReward argument is set to a value different from 0. Depending on its value, the agent may want to stay alive as long as possible or end the episode on a rewarding end state, or even finish episode as fast as possible (suicide).



# Create your own agents

For creating an agent (e.g. a random agent), you must:

- Create a class that inherits from `Agent` (in tp/agents/agent.py), e.g. `RandomAgent`

- Implement the methods act(), observe() and learn() (see the docstrings for more information)

- Add this class and the agent name (e.g. "random") to the dictionary in tp/implemented_agents.py

- In case your agent is Value Based, you may inherit from `ValueBasedAgent` and implement the method `getQValue()` for a visual of your Q Values during training.

You can then train your agent with :
    
```bash
python run.py --agent <your_agent_name>
```
## Implementation tools



# Attribution Information: 
The code used is partially extracted and adapted from the Pacman AI projects.
The Pacman AI projects were developed at UC Berkeley.
Link to the original Berkeley course and Pacman Projects: http://ai.berkeley.edu.
