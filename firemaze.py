import random

class MazeUnit:
    def __init__(self,status,visit):
        self.status = status
        self.visit = visit

class Node:
    def __init__(self, x=None, y=None, parent=None, child=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.child = child


# Prints node list (path from goal to start)
def listPrint(ptr):
    while ptr.parent != None:
        print(ptr.x,ptr.y)
        ptr = ptr.parent
    print(ptr.x,ptr.y)
        
def mazePrint(maze,mazelength):
    for i in range(mazelength):
        print("")
        for j in range(mazelength):
            print(maze[i][j].status,end=" ")
            
def isValid(row, col):  # checks if row col is a valid place to move
    if row >= mazelength or row < 0 or col >= mazelength or col < 0: # is it out of bounds?
        return False
    if maze[row][col].status == "open" and maze[row][col].visit == "no": # is it on fire?
        return True
    return False


# Problem 2: Write DFS algorithm, generate 'obstacle density p' vs 'probability that S can be reached from G' plot
# Recursive DFS execution from (x,y)
def execDFS(maze,x,y):
    node = Node(x,y,None,None)

def DFS(maze,sx,sy,gx,gy):
    # Start at (x,y), check (x+1,y) then (y+1,x) ... want to reach (v,w)
    stack = []
    
   # THIS IS DFS
# def notBFS(maze,curr,gx,gy):
#     # start = Node(sx,sy,None,None)
#     # curr = start
#     # fringe = []
#     if (curr.x = gx and curr.y = gy): # if at target
#         return curr

#     if up = isValid(sx - 1, sy):
#         return BFS(maze,Node(sx - 1, sy,curr,None),gx,gy)
#     if down = isValid(sx + 1, sy):
#         return BFS(maze,Node(sx + 1, sy,curr,None),gx,gy)
#     if left = isValid(sx, sy - 1):
#         return BFS(maze,Node(sx, sy - 1,curr,None),gx,gy)
#     if right = isValid(sx, sy + 1):
#         return BFS(maze,Node(sx, sy + 1,curr,None),gx,gy)

# if path from start node to goal coord then this will send back the node of goal whose parent chain reveals path, if no path then returns None
def BFS(maze, startNode, gx, gy):
    fringe = []
    fringe.append(startNode)
    # while fringe isnt emty
    while len(fringe) != 0:
        curr = fringe.pop(0)
        maze[curr.x][curr.y].visit = "yes"
        # if goal found return the goal, tracking trough parents will give path
        if curr.x == gx and curr.y == gy:
            return curr
        # add all valid neighbors to fringe, up down left right
        if isValid(maze, curr.x - 1, curr.y):
            fringe.append(Node(curr.x - 1, curr.y, curr, None))
        if isValid(maze, curr.x + 1, curr.y):
            fringe.append(Node(curr.x + 1, curr.y, curr, None))
        if isValid(maze, curr.x, curr.y - 1):
            fringe.append(Node(curr.x, curr.y - 1, curr, None))
        if isValid(maze, curr.x, curr.y + 1):
            fringe.append(Node(curr.x, curr.y + 1, curr, None))
    return None

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

maze[0][0].status = "open"  # hardcode top left
maze[mazelength - 1][mazelength - 1].status = "open"  # hardcode bottom right


r1 = Node((0, 0),None)
r2 = Node((1, 0),r1)
listPrint(r2)


# Problem 3: Write BFS and A* algorithms, generate avg '# nodes explored by BFS - # nodes explored by A*' vs 'obstacle density p' plot
