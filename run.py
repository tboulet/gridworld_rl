# ENV
from tp.gridworld_environnement import Env
from tp.parser import parseOptions
# AGENT
from tp.agents.agent import Agent
from tp.implemented_agents import agents_map
# PYTHON
import gym



N_EPISODES = 100

def train(agent : Agent, env : Env, verbose : int = 1):

    for episode in range(N_EPISODES):
        print(f"Episode {episode} starts.")

        state = env.reset()
        done = False
        while not done:
            # Agent takes action
            action = agent.act(state, training = True)

            # Action has effect on environment
            next_state, reward, done, info = env.step(action)

            # Agent observe the transition and possibly learns
            agent.observe(state, action, reward, next_state, done)
            agent.learn()

            # Render environment for user to see
            env.render(agent)

            # Update state
            state = next_state


if __name__ == "__main__":
    
    # Parse options
    options = parseOptions()
    
    # Create the environnement
    env = Env(options)
    
    # Create the agent
    agent_name = options.agent
    agent = agents_map[agent_name](env)
    
    # Train the agent
    train(agent, env)