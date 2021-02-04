import random
import math
import matplotlib.pyplot as plt

class MazeUnit:
    def __init__(self, status, visit):
        self.status = status  # open, bloc, fire
        self.visit = visit  # yes, no, on


class Node:
    def __init__(self, x, y, parent=None, child=None, movesTaken=0, movesLeft=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.child = child
        self.movesTaken = movesTaken
        self.movesLeft = movesLeft


# Prints node list (path from goal to start) #! Modifiable
def listPrint(ptr):
    while ptr.parent != None:
        print(ptr.x, ptr.y)
        ptr = ptr.parent
    print(ptr.x, ptr.y)


def mazePrint(maze, mazelength):
    for i in range(mazelength):
        print("")
        for j in range(mazelength):
            print(maze[i][j].status, end=" ")


def isValid(maze, mazelength, x, y):  # checks if (x,y) is a valid place to move
    if x >= mazelength or x < 0 or y >= mazelength or y < 0:  # is it out of bounds?
        return False
    if maze[x][y].status == "open" and maze[x][y].visit == "no":  # is it on fire?
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


## Problem 2: Write DFS algorithm, generate 'obstacle density p' vs 'probability that S can be reached from G' plot
# DFS execution
def DFS(maze, mazelength, sx, sy, gx, gy):
    # Start at (sx,sy), check (x+1,y) then (y+1,x) ... want to reach (gx,gy)
    startNode = Node(sx, sy, None, None)
    stack = []
    stack.append(startNode)
    prioQ = directionPrio(sx, sy, gx, gy)

    while len(stack) != 0:  # While stack isn't empty
        node = stack.pop()
        maze[node.x][node.y].visit = "yes"
        if node.x == gx and node.y == gy:
            return True
        for i in reversed(
            prioQ
        ):  # Append new nodes to stack in (reverse) order, lower prio appended first
            if i == "r":
                if isValid(maze, mazelength, node.x + 1, node.y):
                    stack.append(Node(node.x + 1, node.y, node, None))
            elif i == "l":
                if isValid(maze, mazelength, node.x - 1, node.y):
                    stack.append(Node(node.x - 1, node.y, node, None))
            elif i == "u":
                if isValid(maze, mazelength, node.x, node.y - 1):
                    stack.append(Node(node.x, node.y - 1, node, None))
            else:
                if isValid(maze, mazelength, node.x, node.y + 1):
                    stack.append(Node(node.x, node.y + 1, node, None))
    return False


## Problem 3: Write BFS and A* algorithms, generate avg '# nodes explored by BFS - # nodes explored by A*' vs 'obstacle density p' plot
# if path from start node to goal coord then this will send back the node of goal whose parent chain reveals path, if no path then returns None
def AstarDist(sx, sy, gx, gy):
    x = gy - sy
    y = gx - sx
    c = math.sqrt(x ** 2 + y ** 2)
    return c


def BFS(maze, startNode, gx, gy):
    fringe = []
    startNode.moves = 0
    fringe.append(startNode)
    maze[startNode.x][startNode.y].visit = "yes"
    # while fringe isnt emty
    while len(fringe) != 0:
        curr = fringe.pop(0)

        # if goal found return the goal, tracking trough parents will give path
        if curr.x == gx and curr.y == gy:
            return curr
        # add all valid neighbors to fringe, up down left right
        up = curr.x - 1
        down = curr.x + 1
        left = curr.y - 1
        right = curr.y + 1
        if isValid(maze, mazelength, up, curr.y):
            fringe.append(
                Node(
                    up,
                    curr.y,
                    curr,
                    None,
                    curr.movesTaken + 1,
                    AstarDist(up, curr.y, gx, gy),
                )
            )
            maze[up][curr.y].visit = "yes"

        if isValid(maze, mazelength, down, curr.y):
            fringe.append(
                Node(
                    down,
                    curr.y,
                    curr,
                    None,
                    curr.movesTaken + 1,
                    AstarDist(down, curr.y, gx, gy),
                )
            )
            maze[down][curr.y].visit = "yes"
        if isValid(maze, mazelength, curr.x, left):
            fringe.append(
                Node(
                    curr.x,
                    left,
                    curr,
                    None,
                    curr.movesTaken + 1,
                    AstarDist(curr.x, left, gx, gy),
                )
            )
            maze[curr.x][left].visit = "yes"
        if isValid(maze, mazelength, curr.x, right):
            fringe.append(
                Node(
                    curr.x,
                    right,
                    curr,
                    None,
                    curr.movesTaken + 1,
                    AstarDist(curr.x, right, gx, gy),
                )
            )
            maze[curr.x][right].visit = "yes"

        fringe.sort(key=lambda Node: (Node.movesTaken + Node.movesLeft))
    # for testing purposes
    # for i in fringe:
    #    print(i.x, i.y, i.movesTaken, i.movesLeft, i.movesTaken + i.movesLeft)
    # print("--------------------")
    return None


## Beginning of main code segment
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

sx = 0
sy = 0
gx = mazelength - 1
gy = mazelength - 1

maze = [[MazeUnit("open", "no") for j in range(mazelength)] for i in range(mazelength)]

for i in range(mazelength):  # fill with obstacles
    for j in range(mazelength):
        if random.random() <= density:
            maze[i][j] = MazeUnit("bloc", "no")

maze[0][0].status = "open"  # hardcode top left to be open
maze[mazelength - 1][mazelength - 1].status = "open"  # hardcode bottom right to be open

## DFS TEST CODE BELOW
# mazePrint(maze,mazelength)
# reachable = DFS(maze,mazelength,sx,sy,gx,gy)
# print("\n")
# print(reachable)
# benton is a lil hoe
# benton is a big hoe