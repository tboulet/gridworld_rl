import optparse

def parseOptions():
    
    optParser = optparse.OptionParser()
    
    # Agent
    optParser.add_option('-a', '--agent',action='store',
                         type='string',dest='agent',default='random',
                         help='Agent to run')

    # Env parameters
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="book",
                         help='Grid to use (case sensitive; options are book, bridge, cliff, maze, default %default)' )
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.0,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
        
    # Graphics
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
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
    opts, args = optParser.parse_args()

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
    # if opts.quiet:
        opts.pause = False
        # opts.manual = False


    return opts
