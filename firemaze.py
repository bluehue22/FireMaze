import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import copy


class MazeUnit:
    def __init__(self, status, visit, q=False):
        self.status = status  # open, bloc, fire
        self.visit = visit  # yes, no, on
        self.q = q


class Node:
    def __init__(self, x, y, parent=None, child=None, movesTaken=0, movesLeft=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.child = child
        self.movesTaken = movesTaken
        self.movesLeft = movesLeft


# Prints node list (path from start to goal)
def listPrint(ptr):
    while (
        ptr.parent != None
    ):  # From goal to start, adjusts all nodes in path to point to correct child
        temp = ptr.parent
        temp.child = ptr
        ptr = ptr.parent
    while ptr.child != None:  # From start to goal, prints node path
        print("({},{})".format(ptr.x, ptr.y), end=" ")
        ptr = ptr.child
    print("({},{})".format(ptr.x, ptr.y))


def mazePrint(maze):
    mazelength = len(maze)
    for i in range(mazelength):
        print("")
        for j in range(mazelength):
            if (
                maze[j][i].status == "open"
            ):  #! Representation of matrix inverted so (x,y) refers to x units across, y units down
                print("O", end="")
            else:
                print("X", end="")


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
    if (
        maze[x][y].status == "open" and maze[x][y].visit == "no" and not maze[x][y].q
    ):  # is it on fire and NOT in fringe?
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
    maze[sx][sy].q = True
    prioQ = directionPrio(sx, sy, gx, gy)

    while len(stack) != 0:  # While stack isn't empty
        node = stack.pop()
        maze[node.x][node.y].visit = "yes"
        if node.x == gx and node.y == gy:
            return node
        for i in reversed(
            prioQ
        ):  # Append new nodes to stack in (reverse) order, lower prio appended first
            if i == "r":
                if isValid(maze, mazelength, node.x + 1, node.y):
                    stack.append(Node(node.x + 1, node.y, node, None))
                    maze[node.x + 1][node.y].q = True
            elif i == "l":
                if isValid(maze, mazelength, node.x - 1, node.y):
                    stack.append(Node(node.x - 1, node.y, node, None))
                    maze[node.x - 1][node.y].q = True
            elif i == "u":
                if isValid(maze, mazelength, node.x, node.y - 1):
                    stack.append(Node(node.x, node.y - 1, node, None))
                    maze[node.x][node.y - 1].q = True
            else:
                if isValid(maze, mazelength, node.x, node.y + 1):
                    stack.append(Node(node.x, node.y + 1, node, None))
                    maze[node.x][node.y + 1].q = True
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
            print(nodes_searched)
            return curr, nodes_searched
        # add all valid neighbors to fringe, up down left right
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
# Fire spreading function, q is flammability rate
def advFire(maze,q):
    mazelength = len(maze)
    mazeCopy = copy.deepcopy(maze)
    for i in range(mazelength):
        for j in range(mazelength):
            if maze[i][j].status == "open": # MazeUnit isn't on fire or blocked
                count = 0 # Check all cells around [i][j] for fire, checking validity of cell location first
                if (i+1)<mazelength and maze[i+1][j].status == "fire":
                    count+=
                if (i-1)>-1 and maze[i-1][j].status == "fire":
                    count+=
                if (j+1)<mazelength and maze[i][j+1].status == "fire":
                    count+=
                if (j-1)>-1 and maze[i][j-1].status =="fire":
                    count+=
                prob = 1-((1-q)**count)
                if random.random() <= prob:
                    mazeCopy[i][j].status = "fire"
    return mazeCopy


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

##ANDREWS A STAR RECHECK
# maze = makeMaze(5, 1)
# A_star_goal_Node, A_star_nodes_searched = A_star(maze, Node(0, 0), 5 - 1, 5 - 1)
# mazePrint(maze)
# print(A_star_nodes_searched)


# PROBLEM 4 CODE SAMPLE
# x = limitTesting(865, 5, "A*") #A*,BSF,DFS
# print("BFS Final Result: {}".format(x))