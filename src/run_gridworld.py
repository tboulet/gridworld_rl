# gridworld.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
import sys
from src.learningAgents import ValueBasedAgent
import src.mdp as mdp
from src.environment import Environment, Grid, Gridworld, GridworldEnvironment

import src.util as util
import optparse



def getUserAction(state, actionFunction):
    """
    Get an action from the user (rather than the agent).

    Used for debugging and lecture demos.
    """
    import src.graphicsUtils as graphicsUtils
    action = None
    while True:
        keys = graphicsUtils.wait_for_keys()
        if 'Up' in keys: action = 'north'
        if 'Down' in keys: action = 'south'
        if 'Left' in keys: action = 'west'
        if 'Right' in keys: action = 'east'
        if 'q' in keys: sys.exit(0)
        if action == None: continue
        break
    actions = actionFunction(state)
    if action not in actions:
        action = actions[0]
    return action

def printString(x): print(x)

def runEpisode(agent, environment, discount, decision, display, message, pause, episode):
    returns = 0
    totalDiscount = 1.0
    environment.reset()
    if 'startEpisode' in dir(agent): agent.startEpisode()
    message("BEGINNING EPISODE: "+str(episode)+"\n")
    while True:

        # DISPLAY CURRENT STATE
        state = environment.getCurrentState()
        
        display(state)
        
        pause()

        # END IF IN A TERMINAL STATE
        actions = environment.getPossibleActions(state)
        if len(actions) == 0:
            message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n")
            return returns

        # GET ACTION (USUALLY FROM AGENT)
        action = decision(state)
        if action == None:
            raise Exception('Error: Agent returned None action')

        # EXECUTE ACTION
        nextState, reward = environment.doAction(action)
        message("Started in state: "+str(state)+
                "\nTook action: "+str(action)+
                "\nEnded in state: "+str(nextState)+
                "\nGot reward: "+str(reward)+"\n")
        # UPDATE LEARNER
        if 'observeTransition' in dir(agent):
            agent.observeTransition(state, action, nextState, reward)

        returns += reward * totalDiscount
        totalDiscount *= discount

    if 'stopEpisode' in dir(agent):
        agent.stopEpisode()

def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="book",
                         help='Grid to use (case sensitive; options are book, bridge, cliff, maze, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\', \'q\', and \'learn\', default %default)')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent')
    optParser.add_option('-v', '--valueSteps',action='store_true' ,default=False,
                         help='Display each step of value iteration')

    opts, args = optParser.parse_args()

    # if opts.manual and (opts.agent != 'qlearning' and opts.agent != 'learn'):
    #     print('## Disabling Agents in Manual Mode (-m) ##')
    #     opts.agent = None

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
    # if opts.quiet:
        opts.pause = False
        # opts.manual = False

    if opts.manual:
        opts.pause = True

    return opts


if __name__ == '__main__':

    opts = parseOptions()

    ###########################
    # GET THE GRIDWORLD
    ###########################

    import src.run_gridworld as run_gridworld
    from grids import getGrid
    mdp = getGrid(opts.grid)
    mdp.setLivingReward(opts.livingReward)
    mdp.setNoise(opts.noise)
    env = GridworldEnvironment(mdp)


    ###########################
    # GET THE DISPLAY ADAPTER
    ###########################

    import src.textGridworldDisplay as textGridworldDisplay
    display = textGridworldDisplay.TextGridworldDisplay(mdp)
    if not opts.textDisplay:
        import src.graphicsGridworldDisplay as graphicsGridworldDisplay
        display = graphicsGridworldDisplay.GraphicsGridworldDisplay(mdp, opts.gridSize, opts.speed)
    try:
        display.start()
    except KeyboardInterrupt:
        sys.exit(0)

    ###########################
    # GET THE AGENT
    ###########################

    from src.implemented_agents import agents_factory_map

    if opts.agent not in agents_factory_map:
        if opts.manual:
            agent = None
        else: 
            raise Exception('Unknown agent type: '+opts.agent)
    else:
        agent_factory = agents_factory_map[opts.agent]
        agent = agent_factory(mdp, opts)

        
    

    ###########################
    # RUN EPISODES
    ###########################
    
    # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
    displayCallback = lambda x: None
    if not opts.quiet:
        if opts.manual and opts.agent == None:
            displayCallback = lambda state: display.displayNullValues(state)
            
        else:
            if opts.agent == "random":
                displayCallback = lambda state: display.displayValues(agent, state, "CURRENT VALUES")
            if isinstance(agent, ValueBasedAgent): 
                displayCallback = lambda state: display.displayQValues(agent, state, "CURRENT Q-VALUES")

    messageCallback = lambda x: printString(x)
    if opts.quiet:
        messageCallback = lambda x: None

    # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
    pauseCallback = lambda : None
    if opts.pause:
        pauseCallback = lambda : display.pause()

    # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)
    if opts.manual:
        decisionCallback = lambda state : getUserAction(state, mdp.getPossibleActions)
    else:
        decisionCallback = agent.getAction

    # RUN EPISODES
    if opts.episodes > 0:
        print()
        print("RUNNING", opts.episodes, "EPISODES")
        print()
    returns = 0
    for episode in range(1, opts.episodes+1):
        
        returns += runEpisode(agent, env, opts.discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
    
    if opts.episodes > 0:
        print()
        print("AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes))
        print()
        print()
    
    # DISPLAY POST-LEARNING VALUES / Q-VALUES
    if isinstance(agent, ValueBasedAgent):
        try:
            display.displayQValues(agent, message = "Q-VALUES AFTER "+str(opts.episodes)+" EPISODES")
            display.pause()
            display.displayValues(agent, message = "VALUES AFTER "+str(opts.episodes)+" EPISODES")
            display.pause()
        except KeyboardInterrupt:
            sys.exit(0)