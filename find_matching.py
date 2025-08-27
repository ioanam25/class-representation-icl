from ortools.graph.python import min_cost_flow
import pickle
import numpy as np
from hungarian_algorithm import algorithm
import hungarian as hun
from collections import deque
import sys
from scipy.optimize import linear_sum_assignment

def labelIt(cost, lx):
    n = len(cost)
    for i in range(n):
        for j in range(n):
            lx[i] = max(lx[i], cost[i][j])

def addTree(x, prevX, inTreeX, prev, slack, slackX, lx, ly, cost):
    inTreeX[x] = True
    prev[x] = prevX
    for y in range(len(slack)):
        if lx[x] + ly[y] - cost[x][y] < slack[y]:
            slack[y] = lx[x] + ly[y] - cost[x][y]
            slackX[y] = x

def updateLabels(inTreeX, inTreeY, slack, lx, ly):
    n = len(slack)
    
    delta = sys.maxsize
    
    for y in range(n):
        if not inTreeY[y]:
            delta = min(delta, slack[y])
    
    for x in range(n):
        if inTreeX[x]:
            lx[x] -= delta
    
    for y in range(n):
        if inTreeY[y]:
            ly[y] += delta
    
    for y in range(n):
        if not inTreeY[y]:
            slack[y] -= delta

def augment(cost, match, inTreeX, inTreeY, prev, xy, yx, slack, slackX, lx, ly):
    
    # augmenting path algorithm
    n = len(cost)
    
    # check if we have found a perfect matching
    if match[0] == n:
        return
    
    x = y = root = 0
    q = deque()
    
    # find root of tree
    for i in range(n):
        if xy[i] == -1:
            root = i
            q.append(root)
            prev[i] = -2
            inTreeX[i] = True
            break
    
    # initialize slack
    for i in range(n):
        slack[i] = lx[root] + ly[i] - cost[root][i]
        slackX[i] = root
    
    # BFS to find augmenting path
    while True:
        
        # building tree with BFS cycle
        while q:
            x = q.popleft()
            
            #iterate through all edges in equality graph
            for y in range(n):
                if lx[x] + ly[y] - cost[x][y] == 0 and not inTreeY[y]:
                    
                    # if y is an exposed vertex in Y
                    # found, so augmenting path exists
                    if yx[y] == -1:
                        x = slackX[y]
                        break
                    else:
                        # else just add y to inTreeY
                        inTreeY[y] = True
                        
                        # add vertex yx[y], which is 
                        # matched with y, to the queue
                        q.append(yx[y])
                        
                        # add edges (x, y) and (y, yx[y]) to the tree
                        addTree(yx[y], x, inTreeX, prev, slack, slackX, lx, ly, cost)
            if y < n:
                break
        
        # augmenting path found
        if y < n:
            break
        
        # else improve labeling
        updateLabels(inTreeX, inTreeY, slack, lx, ly)
        
        for y in range(n):
            if not inTreeY[y] and slack[y] == 0:
                if yx[y] == -1:
                    x = slackX[y]
                    break
                else:
                    inTreeY[y] = True
                    if not inTreeX[yx[y]]:
                        q.append(yx[y])
                        addTree(yx[y], slackX[y], inTreeX, prev, slack, slackX, lx, ly, cost)
        if y < n:
            break
    
    if y < n:
        # augmenting path found
        match[0] += 1
        
        # update xy and yx
        cx = x
        cy = y
        while cx != -2:
            ty = xy[cx]
            xy[cx] = cy
            yx[cy] = cx
            cx = prev[cx]
            cy = ty
        
        # reset inTreeX and inTreeY
        for i in range(n):
            inTreeX[i] = False
            inTreeY[i] = False
        
        # recall function, go to step 1 of the algorithm
        augment(cost, match, inTreeX, inTreeY, prev, xy, yx, slack, slackX, lx, ly)

def findMinCost(cost):
    n = len(cost)
    
    # convert cost matrix to profit matrix
    # by multiplying each element by -1
    for i in range(n):
        for j in range(n):
            cost[i][j] = -1 * cost[i][j]
    
    # to store the results
    result = 0
    
    # number of vertices in current matching
    match = [0]
    
    xy = [-1] * n
    yx = [-1] * n
    lx = [0] * n
    ly = [0] * n
    slack = [0] * n
    slackX = [0] * n
    prev = [0] * n
    
    inTreeX = [False] * n
    inTreeY = [False] * n
    
    labelIt(cost, lx)
    
    augment(cost, match, inTreeX, inTreeY, prev, xy, yx, slack, slackX, lx, ly)

    assignment = []
    for i in range(n):
        result += cost[i][xy[i]]
        assignment.append((i, xy[i]))

    return -1 * result, assignment


if __name__ == "__main__":
    with open('W_all.pkl', 'rb') as f:
        W_all = pickle.load(f)
    # print(W_10)aa
    labels = []
    num_to_label = {}
    num_to_class = {}

    weights = []
    
    for (i, v1) in enumerate(W_all):
        num_to_class[i] = v1
        for (j, v2) in enumerate(W_all[v1]):
            if i == 0:
                labels.append(v2)
            num_to_label[j] = v2

    for (i, v1) in enumerate(W_all):
        weights.append([])
        for (j, v2) in enumerate(W_all[v1]):
            weights[i].append(W_all[v1][v2])

    cost = np.array(weights)
    row_ind, col_ind = linear_sum_assignment(cost, maximize=False)
    print(row_ind, col_ind)
    print(cost[row_ind, col_ind].sum())
    for i in range(len(row_ind)):
        print(num_to_class[row_ind[i]],num_to_label[col_ind[i]])
