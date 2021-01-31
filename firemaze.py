from numpy import zeros as zeros
import random

class MazeUnit:
    def __init__(self,status,visit):
        self.status = status
        self.visit = visit


def mazePrint(maze,mazelength):
    for i in range(mazelength):
        print("")
        for j in range(mazelength):
            print(maze[i][j].status,end=" ")


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

maze = [[MazeUnit("open","no") for j in range(mazelength)] for i in range(mazelength)]

for i in range(mazelength):  # fill with obstacles
    for j in range(mazelength):
        if random.random() <= density:
            maze[i][j] = MazeUnit("bloc","no")

maze[0][0].status = "open"  # hardcode top left
maze[mazelength - 1][mazelength - 1].status = "open"  # hardcode bottom right


# Problem 2: Write DFS algorithm, generate 'obstacle density p' vs 'probability that S can be reached from G' plot
class Node:
    def __init__(self, data=None, parent=None, child = None):
        self.data = data
        self.parent = parent
        self.child = child


class Stack:
    def __init__(self, top):
        self.top = top
    
    def push(node):
        # IN PROGRESS


# Prints node list (path from goal to start)
def listPrint(ptr):
    while ptr.parent != None:
        print(ptr.data)
        ptr = ptr.parent
    print(ptr.data)


# Recursive DFS execution, updating a visitedSet with a node created from nodeCoord
def DFSexecutor(maze,mazelength,visitedSet,stack,nodeCoord):
    node = Node(nodeCoord,None)


def DFS(maze, mazelength, loc1, loc2):
    # Start at (x,y), check (x+1,y) then (y+1,x) ... want to reach (v,w)
    visitedSet = None
    stack = None

r1 = Node((0, 0),None)
r2 = Node((1, 0),r1)
listPrint(r2)


# Problem 3: Write BFS and A* algorithms, generate avg '# nodes explored by BFS - # nodes explored by A*' vs 'obstacle density p' plot
