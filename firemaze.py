import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import copy


class MazeUnit:
    def __init__(self, status, visit, p10=0, p20=0):
        self.status = status  # open, bloc, fire
        self.visit = visit  # yes, no, on
        self.p10 = p10 # first turn in which probability of fire exceeds 10%
        self.p20 = p20 # first turn in which probability of fire exceeds 20%


class Node:
    def __init__(self, x, y, parent=None, child=None, movesTaken=0, movesLeft=0.):
        self.x = x
        self.y = y
        self.parent = parent
        self.child = child
        self.movesTaken = movesTaken
        self.movesLeft = movesLeft


# Prints node list (path from start to goal)
def listPrint(ptr):
    while (ptr.parent != None):  # From goal to start, adjusts all nodes in path to point to correct child
        temp = ptr.parent
        temp.child = ptr
        ptr = ptr.parent
    while ptr.child != None:  # From start to goal, prints node path
        print("({},{})".format(ptr.x, ptr.y), end=" ")
        ptr = ptr.child
    print("({},{})".format(ptr.x, ptr.y))


# returns parent node list (path from start to goal)
def listReverse(ptr):
    while ptr.parent != None:
        temp = ptr.parent
        temp.child = ptr
        ptr = ptr.parent
    return ptr


# prints maze out of easy to read O (open) and X (closed) characters
def mazePrint(maze):
    mazelength = len(maze)
    for i in range(mazelength):
        print()
        for j in range(mazelength):
            if maze[i][j].status == "open":
                if maze[i][j].visit == "on":
                    print("😄", end=" ")
                # elif maze[i][j].visit == "yes":
                #     print("✨", end=" ")
                else:
                    print("🟩", end=" ")
            elif maze[i][j].status == "fire":
                print("🔥", end=" ")
            else:
                print("⬛", end=" ")
    print()


def fringePrint(fringe):
    print("\nFringe: ", end=" ")
    for node in fringe:
        print("({},{})".format(node.x, node.y), end=" ")


# for reverting maze back to unvisited state
def cleanse_maze(maze):
    mazelength = len(maze)
    for i in range(mazelength):
        for j in range(mazelength):
            maze[i][j].visit = "no"
    return maze


###################################################################################################################
## PROBLEM 1: Generate a maze with given dimension and obstacle density
# Return user input values for mazelength and density
def manual_input():
    mazelength = 0
    density = -1

    while mazelength <= 1:
        mazelength = int(input("Enter size length of the maze: "))  # what is size is <1
        if mazelength <= 1:
            print("Number invalid try again")

    while density < 0 or density > 1:
        density = float(input("Enter density: "))
        if density < 0 or density > 1:
            print("Number invalid try again")
    return mazelength, density


# Make a maze with or without user input
def makeMaze(
    mazelength=None, density=None, start_pos: tuple = None, goal_pos: tuple = None
):
    # start coord and goal coord
    if mazelength == None:
        mazelength, density = manual_input()
    if start_pos == None:
        start_pos = (0, 0)
        goal_pos = (mazelength - 1, mazelength - 1)

    sx, sy = start_pos
    gx, gy = goal_pos

    maze = [
        [MazeUnit("open", "no") for j in range(mazelength)] for i in range(mazelength)
    ]

    for i in range(mazelength):  # fill with obstacles
        for j in range(mazelength):
            if random.random() <= density:
                maze[i][j] = MazeUnit("bloc", "no")

    maze[sx][sy].status = "open"  # hardcode top left to be open
    maze[gx][gy].status = "open"  # hardcode bottom right to be open

    return maze


###################################################################################################################
## Problem 2: Write DFS algorithm, generate 'obstacle density p' vs 'probability that S can be reached from G' plot
# Checks if (x,y) is a valid place to move
def isValid(maze, mazelength, x, y):
    if x >= mazelength or x < 0 or y >= mazelength or y < 0:  # is it out of bounds?
        return False
    # is it on fire and NOT in fringe?
    if maze[x][y].status == "open" and maze[x][y].visit == "no":
        return True
    return False


# Returns array of up/down/left/right in priority search order
def directionPrio(sx, sy, gx, gy):
    deltaX = gx - sx
    deltaY = gy - sy
    if deltaX > 0:  # right
        if deltaY > 0:
            if abs(deltaX) > abs(deltaY):
                return ["r", "d", "u", "l"]
            else:
                return ["d", "r", "l", "u"]
        else:
            if abs(deltaX) > abs(deltaY):
                return ["r", "u", "d", "l"]
            else:
                return ["u", "r", "l", "d"]
    else:  # left (and if deltaX = 0)
        if deltaY > 0:
            if abs(deltaX) > abs(deltaY):
                return ["l", "d", "u", "r"]
            else:
                return ["d", "l", "r", "u"]
        else:
            if abs(deltaX) > abs(deltaY):
                return ["l", "u", "d", "r"]
            else:
                return ["u", "l", "r", "d"]


# DFS execution starting at (sx,sy) reaching (gx,gy), returns goal node if success, returns None if not
def DFS(maze, mazelength, sx, sy, gx, gy):
    startNode = Node(sx, sy, None, None)
    stack = []
    stack.append(startNode)
    maze[sx][sy].visit = "yes"

    # Recognize which direction we want to travel farthest and look there first
    prioQ = directionPrio(sx, sy, gx, gy)

    while len(stack) != 0:  # While stack isn't empty
        node = stack.pop()
        if node.x == gx and node.y == gy:
            return node
        for i in reversed(prioQ):  # Append new nodes to stack in (reverse) order, lower prio appended first
            if i == "r":
                if isValid(maze, mazelength, node.x + 1, node.y):
                    stack.append(Node(node.x + 1, node.y, node, None))
                    maze[node.x + 1][node.y].visit = "yes"
            elif i == "l":
                if isValid(maze, mazelength, node.x - 1, node.y):
                    stack.append(Node(node.x - 1, node.y, node, None))
                    maze[node.x - 1][node.y].visit = "yes"
            elif i == "u":
                if isValid(maze, mazelength, node.x, node.y - 1):
                    stack.append(Node(node.x, node.y - 1, node, None))
                    maze[node.x][node.y - 1].visit = "yes"
            else:
                if isValid(maze, mazelength, node.x, node.y + 1):
                    stack.append(Node(node.x, node.y + 1, node, None))
                    maze[node.x][node.y + 1].visit = "yes"
    return None


###################################################################################################################
## Problem 3: Write BFS and A* algorithms, generate avg '# nodes explored by BFS - # nodes explored by A*' vs 'obstacle density p' plot
# if path from start node to goal coord then this will send back the node of goal whose parent chain reveals path, if no path then returns None

# returns direct distance from start coord to end coord
def A_star_Dist(sx, sy, gx, gy):
    x = gy - sy
    y = gx - sx
    c = math.sqrt(x ** 2 + y ** 2)
    return c


# does A star mapping to get to goal coord and returns goal coord Node if path is found whose
# chain of parents will show path and returns None if no path found
def A_star(maze, startNode, gx, gy):
    fringe = []
    startNode.moves = 0
    fringe.append(startNode)
    maze[startNode.x][startNode.y].visit = "yes"
    mazelength = len(maze)
    nodes_searched = 0
    # while fringe isnt emty
    while len(fringe) != 0:
        curr = fringe.pop(0)
        nodes_searched += 1
        # if goal found return the goal, tracking trough parents will give path
        if curr.x == gx and curr.y == gy:
            return curr, nodes_searched
        # add all valid neighbors to fringe, up down left right
        up = curr.x - 1
        down = curr.x + 1
        left = curr.y - 1
        right = curr.y + 1
        # fmt: off
        time_to_sort = 0
    
        if isValid(maze, mazelength, up, curr.y):
            fringe.append(Node(up, curr.y, curr, None, curr.movesTaken + 1, A_star_Dist(up, curr.y, gx, gy)))
            maze[up][curr.y].visit = "yes"
            time_to_sort = 1

        if isValid(maze, mazelength, down, curr.y):
            fringe.append(Node(down, curr.y, curr, None, curr.movesTaken + 1,A_star_Dist(down, curr.y, gx, gy)))
            maze[down][curr.y].visit = "yes"
            time_to_sort = 1

        if isValid(maze, mazelength, curr.x, left):
            fringe.append(Node(curr.x, left, curr, None, curr.movesTaken + 1,A_star_Dist(curr.x, left, gx, gy)))
            maze[curr.x][left].visit = "yes"
            time_to_sort = 1

        if isValid(maze, mazelength, curr.x, right):
            fringe.append(Node(curr.x, right, curr, None, curr.movesTaken + 1,A_star_Dist(curr.x, right, gx, gy)))
            maze[curr.x][right].visit = "yes"
            time_to_sort = 1
        # fmt: on
        if time_to_sort == 1:
            fringe.sort(key=lambda Node: (Node.movesTaken + Node.movesLeft))
        #  for testing purposes
        # print("x y M L   T")
        # for i in fringe:
        #     print(i.x, i.y, i.movesTaken, i.movesLeft, i.movesTaken + i.movesLeft)
        # print("--------------------")

    return None, nodes_searched


def BFS(maze, startNode, gx, gy):
    fringe = []
    fringe.append(startNode)
    mazelength = len(maze)
    maze[startNode.x][startNode.y].visit = "yes"
    nodes_searched = 0
    # while fringe isnt empty
    while len(fringe) != 0:
        curr = fringe.pop(0)

        nodes_searched += 1
        # if goal found return the goal, tracking through parents will give path
        if curr.x == gx and curr.y == gy:
            # print(nodes_searched)
            return curr, nodes_searched
        # add all valid neighbors to fringe, up, down, left, right
        if isValid(maze, mazelength, curr.x - 1, curr.y):
            fringe.append(Node(curr.x - 1, curr.y, curr, None))
            maze[curr.x - 1][curr.y].visit = "yes"

        if isValid(maze, mazelength, curr.x + 1, curr.y):
            fringe.append(Node(curr.x + 1, curr.y, curr, None))
            maze[curr.x + 1][curr.y].visit = "yes"

        if isValid(maze, mazelength, curr.x, curr.y - 1):
            fringe.append(Node(curr.x, curr.y - 1, curr, None))
            maze[curr.x][curr.y - 1].visit = "yes"

        if isValid(maze, mazelength, curr.x, curr.y + 1):
            fringe.append(Node(curr.x, curr.y + 1, curr, None))
            maze[curr.x][curr.y + 1].visit = "yes"

    return None, nodes_searched


###################################################################################################################
## Problem 4: What is largest dimension you can solve DFS, BFS, and A* at p = 0.3 in under a minute?
# Returns True if solvable using DFS (there's a successful path) in <1 min with input dim, p=0.3
def minuteTester(dim, method):
    success = False
    while not success:
        t0 = time.time()
        maze = makeMaze(dim, 0.3)
        if method == "DFS":
            if DFS(maze, dim, 0, 0, dim - 1, dim - 1) != None:
                success = True
        if method == "BFS":
            if BFS(maze, Node(0, 0), dim - 1, dim - 1)[0] != None:
                success = True
        if method == "A*":
            if A_star(maze, Node(0, 0), dim - 1, dim - 1)[0] != None:
                success = True
        t1 = time.time()
        deltaT = t1 - t0
        if success and deltaT < 60:
            print("deltaT: {}".format(deltaT), end=" ")  #!remove
            return True
        elif success and deltaT > 60:
            return False


# Returns largest dimension solvable for search methods, #!Should run >=10 times to ensure maze isn't "free", cannot have any runtime fail
def limitTesting(dim, increment, method):
    while minuteTester(dim, method):
        print("dim: {}".format(dim))  #! removed
        dim += increment
    return dim - increment


###################################################################################################################
## Fire maze functions and strategies
# Fire spreading function, q is flammability rate,
def advFire(maze, q):
    mazelength = len(maze)
    mazeCopy = copy.deepcopy(maze)
    for i in range(mazelength):
        for j in range(mazelength):
            if maze[i][j].status == "open":  # MazeUnit isn't on fire or blocked
                count = 0  # Check all cells around [i][j] for fire, checking validity of cell location first
                if (i + 1) < mazelength and maze[i + 1][j].status == "fire":
                    count += 1
                if (i - 1) > -1 and maze[i - 1][j].status == "fire":
                    count += 1
                if (j + 1) < mazelength and maze[i][j + 1].status == "fire":
                    count += 1
                if (j - 1) > -1 and maze[i][j - 1].status == "fire":
                    count += 1
                prob = 1 - ((1 - q) ** count)
                if random.random() <= prob:
                    mazeCopy[i][j].status = "fire"
    return mazeCopy


# returns a valid fire maze, will not work if density too high for path to exist
def makeFireMaze(mazelength, density):
    while True:
        maze = makeMaze(mazelength, density)
        # can I get to the end
        if DFS(maze, len(maze), 0, 0, len(maze) - 1, len(maze) - 1) != None:
            break

    while True:
        maze = cleanse_maze(maze)
        x = random.randint(0, len(maze) - 1)
        y = random.randint(0, len(maze) - 1)
        # place fire on open square
        if maze[x][y].status == "open":
            # fire cant not start at end or start
            if not (x == y and x == 0) and not (x == y and x == len(maze) - 1):
                # check if fire can reach player
                if DFS(maze, len(maze), 0, 0, x, y) != None:
                    maze[x][y].status = "fire"
                    maze = cleanse_maze(maze)
                    # can you still make it out
                    if DFS(maze, len(maze), 0, 0, len(maze) - 1, len(maze) - 1) != None:
                        maze = cleanse_maze(maze)
                        return maze
                    maze[x][y].status = "open"


# send valid fire maze will return (true and goal node) or (False and last node of fail)
def strat1(maze, q):
    ptr, _ = BFS(maze, Node(0, 0), len(maze) - 1, len(maze) - 1)
    maze = cleanse_maze(maze)
    ptr = listReverse(ptr)
    while ptr.child != None:
        maze[ptr.x][ptr.y].visit = "yes"
        ptr = ptr.child

        maze = advFire(maze, q)

        if isValid(maze, len(maze), ptr.x, ptr.y) == False:
            print("we failed going to", ptr.x, ptr.y)  # benton
            print("-----------------")  # benton
            mazePrint(maze)  # benton
            print("-----------------")  # benton
            return False, ptr
        maze[ptr.x][ptr.y].visit = "on"
        print("-----------------")  # benton
        mazePrint(maze)  # benton
        print("-----------------")  # benton
    if ptr.x == len(maze) - 1 and ptr.y == len(maze) - 1:
        return True, ptr
    return False, ptr


# send valid fire maze will return (true and goal node) or (False and last node of fail)
def strat2(maze, q):
    ptr, _ = BFS(maze, Node(0, 0), len(maze) - 1, len(maze) - 1)
    maze = cleanse_maze(maze)
    ptr = listReverse(ptr)
    answer = Node(ptr.x, ptr.y)
    # returns path taken
    while True:
        print("start", ptr.x, ptr.y)
        ptr = ptr.child
        print("child", ptr.x, ptr.y)
        maze = advFire(maze, q)
        answer.child = Node(ptr.x, ptr.y, answer)
        answer = answer.child
        if isValid(maze, len(maze), ptr.x, ptr.y) == False:
            print("we failed going to", ptr.x, ptr.y)  # benton
            print("-----------------")  # benton
            mazePrint(maze)  # benton
            print("-----------------")  # benton
            return False, answer
        print("-----------------")  # benton
        maze[ptr.x][ptr.y].visit = "on"
        mazePrint(maze)  # benton
        maze[ptr.x][ptr.y].visit = "no"
        print("-----------------")  # benton
        ptr, _ = BFS(maze, Node(ptr.x, ptr.y), len(maze) - 1, len(maze) - 1)

        maze = cleanse_maze(maze)
        if ptr == None:
            print("There is no path anymore")
            return False, answer
        ptr = listReverse(ptr)
        if ptr.x == len(maze) - 1 and ptr.y == len(maze) - 1:
            return True, answer
    # return False, answer
    # ptr = BFS(maze, Node(0, 0), len(maze) - 1, len(maze) - 1)[0]
    # ptr = listReverse(ptr)
    # answer = Node(ptr.x, ptr.y)  # returns path taken

    # while ptr.child != None:
    #     ptr = ptr.child
    #     maze = advFire(maze, q)
    #     answer.child = Node(ptr.x, ptr.y, answer)
    #     answer = answer.child
    #     if isValid(maze, len(maze), ptr.x, ptr.y) == False:
    #         return False, answer
    #     ptr = BFS(maze, Node(ptr.x, ptr.y), len(maze) - 1, len(maze) - 1)[0]
    #     if ptr == None:
    #         return False, answer
    #     ptr = listReverse(ptr)
    # return True, answer


# Checks if fire is within n blocks (manhattan distance)
def checkFire(maze, currx, curry, n):
    mazelength = len(maze)
    for i in range(currx - n, currx + n + 1):
        for j in range(curry - (n - abs(currx-i)), curry + (n - abs(currx-i)) + 1):
            if (i > -1 and i < mazelength) and (j > -1 and j < mazelength):
                if maze[i][j].status == "fire":
                    return True
    return False


# fmt: off
# Gives probability of a target MazeUnit being on fire given probabilities a,b,c,d of the surrounding squares being on fire (q is fire spreading rate)
def fireProb(q, a, b, c, d):
    out = float(q * (a*(1-b)*(1-c)*(1-d) + (1-a)*b*(1-c)*(1-d) + (1-a)*(1-b)*c*(1-d) + (1-a)*(1-b)*(1-c)*d)
        + (1-(1-q)**2) * (a*b*(1-c)*(1-d) + a*(1-b)*c*(1-d) + a*(1-b)*(1-c)*d + (1-a)*b*c*(1-d) + (1-a)*b*(1-c)*d + (1-a)*(1-b)*c*d)
        + (1-(1-q)**3) * (a*b*c*(1-d) + a*b*(1-c)*d + a*(1-b)*c*d + (1-a)*b*c*d)
        + (1-(1-q)**4) * a*b*c*d)
    return out


# Sets fire probabilities and t20 values within m blocks of (currx,curry)
def setFireProb(q, maze, currx, curry, m):
    probMatrix = np.zeros((2*m + 1, 2*m + 1))  # Matrix that represents the probability region around our agent
    offsetx, offsety = (currx - m,curry - m)  # (0,0) of probMatrix corresponds to top left corner of maze area
    
    # Initialize fires in probability matrix
    for i in range(currx - m, currx + m + 1):
        for j in range(curry - m, curry + m + 1):
            if (i > -1 and i < len(maze)) and (j > -1 and j < len(maze)):
                if maze[i][j].status == "fire":
                    probMatrix[i - offsetx][j - offsety] = 1.0
    
    # Update probability matrix with new values every turn and sets all unset values by the end to max turns
    for turns in range(m):
        updateMatrix = copy.deepcopy(probMatrix)
        for i in range(currx - m, currx + m + 1):
            for j in range(curry - m, curry + m + 1):
                if (i > -1 and i < len(maze)) and (j > -1 and j < len(maze)):
                    if maze[i][j].status == "open":
                        # Check surrounding squares for probabilities (within range of probMatrix and maze probability region)
                        a, b, c, d = 0.0, 0.0, 0.0, 0.0 # Default adjacent square probability values are 0, this is updated to any present probabilities
                        
                        if ((i + 1) < len(maze) and (i + 1 - offsetx) < (2*m + 1) and
                            (maze[i + 1][j].status == "open" or maze[i + 1][j].status == "fire")
                        ):
                            a = probMatrix[i + 1 - offsetx][j - offsety]
                        
                        if ((i - 1) > -1 and
                            (maze[i - 1][j].status == "open" or maze[i - 1][j].status == "fire")
                        ):
                            b = probMatrix[i - 1 - offsetx][j - offsety]
                        
                        if ((j + 1) < len(maze) and (j + 1 - offsety) < (2*m + 1) and
                            (maze[i][j + 1].status == "open" or maze[i][j + 1].status == "fire")
                        ):
                            c = probMatrix[i - offsetx][j + 1 - offsety]
                        
                        if ((j - 1) > -1 and 
                            (maze[i][j - 1].status == "open" or maze[i][j - 1].status == "fire")
                        ):
                            d = probMatrix[i - offsetx][j - 1 - offsety]
                        
                        # Calculate probability of Maze(i,j) to be on fire
                        if a != 0 or b != 0 or c != 0 or d != 0:
                            updateMatrix[i - offsetx][j - offsety] = fireProb(q, a, b, c, d)
                        
                        # Update p10 and p20 values
                        if updateMatrix[i - offsetx][j - offsety] > 0.1 and maze[i][j].p10 == 0:
                            maze[i][j].p10 = turns
                        if updateMatrix[i - offsetx][j - offsety] > 0.2 and maze[i][j].p20 == 0:
                            maze[i][j].p20 = turns

                        # Default p10 and p20 values to max number of turns taken if not filled by last turn
                        if turns == (m - 1) and maze[i][j].p10 == 0:
                            maze[i][j].p10 = turns
                        if turns == (m - 1) and maze[i][j].p20 == 0:
                            maze[i][j].p20 = turns
        
        probMatrix = updateMatrix
    return


def clearFireProb(maze, currx, curry, m):
    for i in range(currx - m, currx + m + 1):
        for j in range(curry - m, curry + m + 1):
            if (i > -1 and i < len(maze)) and (j > -1 and j < len(maze)):
                maze[i][j].p10 = 0
                maze[i][j].p20 = 0


# Return a specific rating (under 10%, under 20%, or above 20% chance of fire)
def safetyRating(maze,node):
    x = node.x
    y = node.y
    if maze[x][y].p10 == 0:
        return 0
    elif node.movesTaken < maze[x][y].p10: # Best safety rating
        return 0
    elif node.movesTaken < maze[x][y].p20:
        return 1
    else: # Worst safety rating
        return 2


# Sort through fringe searching for insertion point
def safetyInsert(maze,fringe,node):
    ptr = 0
    nodeRating = safetyRating(maze,node)

    while ptr < len(fringe):
        ptrRating = safetyRating(maze,fringe[ptr])

        if nodeRating == ptrRating:
            if node.movesLeft > fringe[ptr].movesLeft:
                ptr += 1
            else:
                break
        elif nodeRating > ptrRating: # Better node rating than ptr's
            ptr += 1
        else: # looking too far down fringe
            break
                
    # insert node i at ptr + 1 location
    if ptr == len(fringe):
        fringe.append(node)
    else:
        fringe.insert(ptr,node)


# Sorts nodes into fringe based on safety first, then distance to goal
def safetyFirst(maze, sx, sy, gx, gy):
    startNode = Node(sx, sy, None, None) # startNode.movesTaken = 0 by default
    fringe = []
    fringe.append(startNode)
    maze[sx][sy].visit = "yes"

    while len(fringe) != 0: # While fringe isn't empty
        node = fringe.pop(0)
        if node.x == gx and node.y == gy:
            return node

        # Add new (adjacent to current position) nodes to fringe in safety priority & distance order
        if isValid(maze, len(maze), node.x + 1, node.y):
            tempNode = Node(node.x + 1, node.y, node, None, node.movesTaken + 1, A_star_Dist(node.x + 1, node.y, gx, gy))
            safetyInsert(maze, fringe, tempNode)
            maze[node.x + 1][node.y].visit = "yes"
        
        if isValid(maze, len(maze), node.x, node.y + 1):
            tempNode = Node(node.x, node.y + 1, node, None, node.movesTaken + 1, A_star_Dist(node.x, node.y + 1, gx, gy))
            safetyInsert(maze, fringe, tempNode)
            maze[node.x][node.y + 1].visit = "yes"
        
        if isValid(maze, len(maze), node.x - 1, node.y):
            tempNode = Node(node.x - 1, node.y, node, None, node.movesTaken + 1, A_star_Dist(node.x - 1, node.y, gx, gy))
            safetyInsert(maze, fringe, tempNode)
            maze[node.x - 1][node.y].visit = "yes"
        
        if isValid(maze, len(maze), node.x, node.y - 1):
            tempNode = Node(node.x, node.y - 1, node, None, node.movesTaken + 1, A_star_Dist(node.x, node.y - 1, gx, gy))
            safetyInsert(maze, fringe, tempNode)
            maze[node.x][node.y - 1].visit = "yes"

    return None


# Return the goal node
def strat3(maze, q):
    path = []
    mazelength = len(maze)
    agent = Node(0, 0)
    maze[0][0].visit = "on" # Agent starts at (0,0)
    
    # Use BFS to establish an initial "guess" path
    ptr, _ = BFS(maze, Node(0, 0), mazelength - 1, mazelength - 1)
    maze = cleanse_maze(maze)
    # Traverses up from goal node produced by BFS, adding to path in reverse
    while ptr.parent != None:
        path.insert(0, ptr)
        ptr = ptr.parent

    # While our agent isn't dead
    n = 3
    m = n + 2
    while maze[agent.x][agent.y].status != "fire":
        # If person gets too close to fire (x blocks)
        # 	Recalc like start 2, cleanse maze!
        # 	Do strat with start 1 pathing
        if checkFire(maze, agent.x, agent.y, n):
            maze = cleanse_maze(maze)
            setFireProb(q,maze,agent.x,agent.y,m)

            # Add nodes to path by probability, then dist.
            ptr = safetyFirst(maze, agent.x, agent.y, mazelength - 1, mazelength - 1)
            clearFireProb(maze,agent.x,agent.y,m)
            
            if ptr != None: # Update path
                path = []
                while ptr.parent != None:
                    path.insert(0,ptr)
                    ptr = ptr.parent
            else: # There is no path to goal, thus agent will die
                return False


        # Move to next spot on path and check for goal
        newLoc = path.pop(0)
        if (newLoc.x == mazelength - 1) and (newLoc.y == mazelength - 1):
            return True
        maze[agent.x][agent.y].visit = "yes" #! Can remove
        agent.x = newLoc.x
        agent.y = newLoc.y
        maze[agent.x][agent.y].visit = "on" #! Can remove

        # Advance fire
        maze = advFire(maze, q)
    
    # If we exit while loop above, agent is dead
    return False
# fmt: on

###################################################################################################################
## Beginning of main code segment

## DFS TEST CODE SAMPLE, prints path if possible
# sx,sy,gx,gy,mazelength,p = 0,0,3,3,4,0.25
# maze = makeMaze(mazelength,p)
# mazePrint(maze)
# reachable = DFS(maze,mazelength,sx,sy,gx,gy)
# if reachable == None:
#     print("\nNot possible to reach goal from start")
# else:
#     print("\nListprint:",end=" ")
#     listPrint(reachable)


## DFS PROBABILITY VS OBJECT DENSITY PLOT CODE
# probArr = []
# sx,sy,mazelength,numTrials,plotPoints = 0,0,100,1000,101 # Set parameters here
# gx = gy = mazelength - 1
# for i in np.linspace(0,1,plotPoints): # Object density array
#     successes = 0
#     for j in range(numTrials): # trials per object density value
#         maze = makeMaze(mazelength,i)
#         if DFS(maze,mazelength,sx,sy,gx,gy):
#             successes += 1
#     probArr.append(float(successes/numTrials)) # Add probability point to array

# plt.plot(np.linspace(0,1,plotPoints),probArr)
# plt.xlabel("Obstacle Density")
# plt.ylabel("Probability of Success")
# plt.show()


# ## BFS / A* problem 3 CODE SAMPLE
# i = 750
# density = 0
# attempts_to_solve = 0
# density_list = []
# A_star_num_node_list = []
# BFS_num_node_list = []
# while density <= 1:
#     maze = makeMaze(i, density)
#     BFS_goal_Node, BFS_nodes_searched = BFS(maze, Node(0, 0), i - 1, i - 1)
#     maze = cleanse_maze(maze)
#     A_star_goal_Node, A_star_nodes_searched = A_star(maze, Node(0, 0), i - 1, i - 1)
#     # mazePrint(maze)
#     if A_star_goal_Node != None:
#         # listPrint(A_star_goal_Node)
#         # print("A-yes", A_star_nodes_searched, BFS_num_node_list)
#         A_star_num_node_list.append(A_star_nodes_searched)
#         BFS_num_node_list.append(BFS_nodes_searched)
#         density_list.append(density)
#         density = density + 0.01
#         print(density)
#         attempts_to_solve = 0
#     else:
#         attempts_to_solve += 1
#     if attempts_to_solve == 25:
#         A_star_num_node_list.append(A_star_nodes_searched)
#         BFS_num_node_list.append(BFS_nodes_searched)
#         density_list.append(density)
#         density = density + 0.01
#         print(density)
#         attempts_to_solve = 0
# plt.xlabel("Obstacle Density")
# plt.ylabel("Nodes Checked")
# plt.plot(density_list, BFS_num_node_list, color="red", label="BFS")
# plt.plot(density_list, A_star_num_node_list, color="blue", label="A*")
# plt.legend(loc="best")
# plt.show()

## ANDREWS A STAR RECHECK
# maze = makeMaze(5, 1)
# A_star_goal_Node, A_star_nodes_searched = A_star(maze, Node(0, 0), 5 - 1, 5 - 1)
# mazePrint(maze)
# print(A_star_nodes_searched)


# PROBLEM 4 CODE SAMPLE
# x = limitTesting(6000, 100, "DFS")  # A*,BSF,DFS
# print("BFS Final Result: {}".format(x))

## Strat 1 and 2 test code
# mazelength = 10
# density = 0.3
# q = 0.2
# maze = makeFireMaze(mazelength, density)
# print("first maze")
# mazePrint(maze)

# end, _ = BFS(maze, Node(0, 0), mazelength - 1, mazelength - 1)

# path, end = strat2(maze, 0.2)
# if path:
#     listPrint(end)
#     print("we did it")
# else:
#     listPrint(end)
#     print("we died")
# mazePrint(maze)

## Strat 1,2,3 plot code
mazelength = 200
density = 0.3
successProb = []
for q in range(0, 1, 0.01):
    successes = 0
    #! WE NEED TO RESET FIRE 10X PER MAZE
    for i in range(1000): # 1000 trials per q value
        maze = makeFireMaze(mazelength, density)
        if strat3(maze, q):
            successes += 1
    qProb = float(successes/1000)
    successProb.append(qProb)