from tp.environnement import Env
from tp.agents import agents_factory_map

from argparse import ArgumentParser

N_EPISODES = 100

def train(agent, env):

    for episode in range(N_EPISODES):

        state = env.reset()
        done = False
        while not done:
            # Agent takes action
            action = agent.act(state, training = True)
            # Action has effect on environment
            next_state, reward, done, _ = env.step(action)
            # Agent learns from the experience (this have to be seen as very large definition)
            agent.learn(state, action, reward, next_state, done)

            # Render environment for user to see
            env.render(agent) 
            state = next_state


if __name__ == "__main__":
    # Get args
    parser = ArgumentParser(description="Run a reinforcement learning agent")
    parser.add_argument("--agent", type=str, required=True, help="Agent to run", choices = agents_factory_map.keys())

    args = parser.parse_args()
    agent_name = args.agent

    # Create the environnement
    env = Env()
    # Create the agent
    agent = agents_factory_map[agent_name]
    # Run the agent
    train(agent, env)