
from src.environment import Gridworld, makeGrid


def getGrid(grid_name : str):
    if grid_name == 'cliff':
        return getCliffGrid()
    elif grid_name == 'cliff2':
        return getCliffGrid2()
    elif grid_name == 'discount':
        return getDiscountGrid()
    elif grid_name == 'bridge':
        return getBridgeGrid()
    elif grid_name == 'book':
        return getBookGrid()
    elif grid_name == 'maze':
        return getMazeGrid()
    else:
        raise ValueError('Unknown grid name: ' + grid_name)



### GRID DEFINITIONS ###

def getCliffGrid():
    grid = [[' ',' ',' ',' ',' '],
            ['S',' ',' ',' ',10],
            [-100,-100, -100, -100, -100]]
    return Gridworld(makeGrid(grid))

def getCliffGrid2():
    grid = [[' ',' ',' ',' ',' '],
            [8,'S',' ',' ',10],
            [-100,-100, -100, -100, -100]]
    return Gridworld(grid)

def getDiscountGrid():
    grid = [[' ',' ',' ',' ',' '],
            [' ','#',' ',' ',' '],
            [' ','#', 1,'#', 10],
            ['S',' ',' ',' ',' '],
            [-10,-10, -10, -10, -10]]
    return Gridworld(grid)

def getBridgeGrid():
    grid = [[ '#',-100, -100, -100, -100, -100, '#'],
            [   1, 'S',  ' ',  ' ',  ' ',  ' ',  10],
            [ '#',-100, -100, -100, -100, -100, '#']]
    return Gridworld(grid)

def getBookGrid():
    grid = [[' ',' ',' ',+1],
            [' ','#',' ',-1],
            ['S',' ',' ',' ']]
    return Gridworld(grid)

def getMazeGrid():
    grid = [[' ',' ',' ',+1],
            ['#','#',' ','#'],
            [' ','#',' ',' '],
            [' ','#','#',' '],
            ['S',' ',' ',' ']]
    return Gridworld(grid)
