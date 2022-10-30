###################################################################################################################
# Enumerating graphs in R(K4,J5,18) by adding edges between R(K3, J5, 7) and R(K4, J4, 7)
###################################################################################################################
import numpy as np
from ortools.sat.python import cp_model
from itertools import chain, combinations
import copy
import time

###################################################################################################################
# Some printing functions (who might instead return convenient strings)

# Print binary form of decimal number
def printBit(decimal, n):
    return bin(int(decimal))[2:].zfill(n)

# Print binary form of list of decimal numbers
def printBitList(decimalList, n):
    return [printBit(decimal, n) for decimal in decimalList]

# Print rows of a adjacency matrix on seperate lines
def printGraph(graph):
     return "\n".join(str(row) for row in graph) + "\n"

# Print a collection of adjacency matrixes
def printGraphs(graphs):
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
    for graph in graphs:
        print(printGraph(graph))

# Print the key value pairs of a dictionary on seperate lines.
def printDict(dict):
    return "\n".join(str(key) + ": " + str(dict[key]) for key in dict) + "\n"
###################################################################################################################

###################################################################################################################
# Decoding and Formating

# Decode graphs in g6 format to their adjacency matrixes,
# To determine the number of vertices in the graph, subtract 63 from the first byte.
# Take each subsequent byte and subtract 63, making sure to pad the resulting sequence.
# Then string all the segments together, remembering to cut off the sequence so its length is n(n-1)/2

def decodeG6(compressed, isSquare): # Option for triangle and square adjacency matrix
    n = ord(compressed[0]) - 63 # number of vertices
    length = int(n*(n - 1)/2) # number of 1's and 0's representing upper triangle of adjacency matrix

    bitVect = "" # Our bit vector will be represented by a string
    for character in compressed[1:]:
        bitVect += bin(ord(character) - 63)[2:].zfill(6) # subtract 63 from each byte and string the information together
    bitVect = bitVect[:length] # make sure our bit vector is the correct length

    if not isSquare:
        return bitVect

    adjacencyMatrix = [[0 for i in range(n)] for j in range(n)] # Our adjacency matrix will be a list of lists
    index = 0
    for column in range(1, n):
        for row in range(column):
            adjacencyMatrix[row][column] = adjacencyMatrix[column][row] = int(bitVect[index]) # We iterate through the rows and the columns, filling them out as we go
            index += 1
    return adjacencyMatrix

def formatBitVect(bitVect): # Takes string bit vector and converts it into a int representing the bit vector (remember to keep track of length)
    return int(bitVect, 2)

def formatAdjacencyMatrix(adjacencyMatrix):# Takes lists of lists adjacency matrix and converts it into a list of ints (also remember to keep track of length)
    return [int(''.join(str(ele) for ele in row), 2) for row in adjacencyMatrix]

###################################################################################################################

###################################################################################################################
# Find feasible cones or subsets of the original graph that do not contain triangles and whose complements do not contain independent 3-sets.

# First, an algorithm to find all maximal cliques (or independent sets).  This will help find all triangles in the R(K4, J4) (or independent 3-sets)
# For more information about the algorithm: https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
# We will represent subsets of vertices as 1's located at proper indexes
# n will always be the number of vertices

def bitIndex(bitVect, i): # Find ith digit, starting from the right
    return (bitVect & (1 << i)) >> i

# A helper function that turns a bit form of a set into a list of bit form elements
def expand(bitSet, n):
    return [1 << i for i in range(n) if bitIndex(bitSet, i)] # This will return a list of bit form elements found only in the set

# We initialize the recursion by having R and X = 0, while P will be the set of all vertices, or 111...
def maxClique(R, P, X, maxCliques, adjacencyMatrix, n): # Our cliques will recursively include all the vertices in R, some of the vertices in P, and none of the vertices in X
    if P == 0 and X == 0:
        maxCliques.append(R) # We will store our resulting cliques, in the form of int bit vectors, in a list
        return None
    u = n - (P | X).bit_length() # We choose a pivot vertex
    for v in expand(P & ~adjacencyMatrix[u], n): # We iterate through the vertices in P we need to check
        neighbors = adjacencyMatrix[n - v.bit_length()] # This will index the adjacency matrix at the correct point
        maxClique(R | v, P & neighbors, X & neighbors, maxCliques, adjacencyMatrix, n) # We recurse, restricting our focus to v's neighbors
        # v is not available anymore
        P = P & ~v
        X = X | v

# A helper function to take the complement of a formatted adjacency matrix
def complement(adjacencyMatrix, n):
    flip = (1 << n) - 1
    return [flip ^ adjacencyMatrix[row] ^ (1 << n - row - 1) for row in range(n)]

# To find independent sets, we look instead at the complement adjacency matrix
def maxSet(R, P, X, maxSets, adjacencyMatrix, n):
    compMatrix = complement(adjacencyMatrix, n)
    return maxClique(R, P, X, maxSets, compMatrix, n)

# Class needed to "print" all solutions
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_vector = []
    def on_solution_callback(self):
        solution = 0
        for v in self.__variables:
            solution = solution << 1
            solution += self.Value(v)
        self.__solution_vector.append(solution)
    def solution_vector(self): # Solution vector, tells us the values of the variables
        return self.__solution_vector

# The feasible cones algorithm itself!
def feasibleCones(H, n):
    vertices = range(n)
    model = cp_model.CpModel() # Create model
    variables = [model.NewBoolVar('v'+str(i)) for i in vertices] # Make variables for each of the vertices determining if they will be present in the feasible cone

    # Add triangle conditions
    triangles = []
    maxClique(0, (1 << n) - 1, 0, triangles, H, n)
    triangles = [triangle for triangle in triangles if bin(triangle).count('1') == 3] # We only look at maximal cliques of three
    for triangle in triangles:
        model.AddBoolOr([variables[vertex].Not() for vertex in vertices if bitIndex(triangle, n - vertex - 1)]) # We ensure that a feasible cone doesn't contain all three vertices

    # Add complement triangle conditions
    ind3sets = []
    maxSet(0, (1 << n) - 1, 0, ind3sets, H, n)
    ind3sets = [set for set in ind3sets if bin(set).count('1') == 3]
    for set in ind3sets:
        model.AddBoolOr([variables[vertex] for vertex in vertices if bitIndex(set, n - vertex - 1)]) # We ensure that a feasible cone is connected to the independent 3-set

    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(variables)
    solver.parameters.enumerate_all_solutions = True # We want all solutions
    status = solver.Solve(model, solution_printer)
    bitSolutions = solution_printer.solution_vector()
    return bitSolutions
###################################################################################################################

###################################################################################################################
# Currently, we only have algorithms to find maximal cliques and independent sets.  We wish to create a more complete repertoire of clique and subgraph finding functions.

# It will be useful to procure a list of edges and non-edges directly from an adjacency matrix
def findEdges(adjacencyMatrix, n): # iterates over list of list adjacency matrices
    edgeSet = [] # we keep the edges in a set so we can easily search within them later
    nonEdgeSet = []
    for row in range(n - 1):
        for column in range(row + 1, n): # iterates through rows and columns we need to check
            if adjacencyMatrix[row][column] == 1:
                edgeSet.append( (1 << (n - row - 1)) | (1 << (n - column - 1)) )
            else:
                nonEdgeSet.append( (1 << (n - row - 1)) | (1 << (n  - column - 1)) )
    return edgeSet, nonEdgeSet

# It will also be useful to generate cliques or independent subgraphs of order - 1 from lists of maximal subgraphs  of order
# Helper function to find the order subgraphs of order - 1 found in a subgraph of order
def oneFindSmaller(maximal, n):
    return {maximal & ~(1 << i) for i in range(n) if bitIndex(maximal, i) == 1}

# We then can define our original function
def findSmaller(maximalList, order, n): # order is the order of the maximal subgraph
    smaller = []
    for maximal in maximalList: # We iterate through the list of maximal subgraphs
        size = bin(maximal).count('1')
        if size == order - 1:
            smaller.append(maximal) # We add any subgraph of already the correct size
        elif size == order:
             for i in range(n): # We break up the bigger maximal subgraphs
                sub = maximal & ~(1 << i)  # One possible smaller clique
                if bitIndex(maximal, i) == 1 and sub not in smaller: # We make sure there are no duplicates
                    smaller.append(sub)
    return smaller

# Using independent sets of order n and order - 1, we can find all subgraphs Jorder's.
# If there are two subgraphs of order - 1 sharing order - 2 vertices, and if the induced subgraph on the union of the sets of vertices
# of these subgraphs is not found in the set of order independent sets, then we have found a Jorder.
def findJ(smaller, larger, order): # We take order - 1 and order independent sets.
    j = []
    length = len(smaller)
    for k in range(length - 1):
        for l in range(k + 1, length): # We iterate through all pairs in smaller
            first, second = smaller[k], smaller[l]
            if bin(first & second).count('1') == order - 2: # We check the intersection on the set of their vertices
                combo = first | second
                if combo not in larger: # We check if the union on their sets of their vertices is in larger
                    j.append(combo)
    return j

# Finds the maximum graphs in a list of maximal subgraphs
def maximalToMaximum(maximal, order):
    return [sub for sub in maximal if bin(sub).count('1') == order]

###################################################################################################################
# We now define a class for the H graphs, which are in R(K4, J4, 10)
# We calculate its cliques and independent sets of order 2 and 3, as well as its independent J3s
# Using these, we define collapsing rules

# But first, a helper function to see if a one set in bit form contains another
def isContains(bitVect, bitVect2):
    return (bitVect1 & bitVect2) == bitVect2

# Helper function to see if the intersection of one set with another has at least one element
def isNotConnect(bitVect1, bitVect2):
    return (bitVect1 & bitVect2) == 0

# Helper function to check if at least element of a set is contains in the intersection sets or if two elements are contained in the union of the two sets
def isNotDoubleConnect(intersection, union, bitVect):
    return not (isNotConnect(intersection, bitVect) or (bin(union & bitVect).count('1') >= 2))

# Special, faster case when the the set has only two elements
def isNotDoubleConnect2(intersection, union, bitVect):
    return not (isNotConnect(intersection, nonEdge) or isContains(union, nonEdge))

class H(object):
    def __init__(self, adjacencyMatrix, n):
        self.n = n # Number of vertices

        self.k2, self.e2 = findEdges(adjacencyMatrix, n) # Find edges and non-edges
        formattedAdjacencyMatrix = formatAdjacencyMatrix(adjacencyMatrix)

        k3 = []
        maxClique(0, (1 << n) - 1, 0, k3, formattedAdjacencyMatrix, n)
        self.k3 = maximalToMaximum(k3, 3) # Find triangles

        e3 = []
        maxSet(0, (1 << n) - 1, 0, k3, formattedAdjacencyMatrix, n)
        self.e3 = maximalToMaximum(e3, 3) # Find independent 3-sets

        self.j3 = findJ(self.e2, self.e3, 3)

    def K2(cone1, cone2): # Collapsing rule if two vertices in G are connected
        intersection = cone1 & cone2 # We look at the intersection of the neighborhoods of the cones
        for edge in self.k2:
            if isContains(union, edge): # We make sure that the it contains no edges
                return False

        union = cone1 | cone2 # We look at the union of the neighborhoods of the cones
        for ind3set in self.e3:
            if isNotConnect(union, in3set): # We make sure it connects to every indpendent 3-set
                return False
        return True

    def E2(cone1, cone2): # Collapsing rule if two vertices in G are not connected
        union = cone1 | cone2 # We look at the union of the neighborhoods of the cones
        for sub in self.j3:
            if isNotConnect(union, sub): # We make sure it connects to every independent j3
                return False

        intersection = cone1 & cone2 # We also need to look at the intersection of the two neighborhoods
        for ind3set in self.e3:
            if isNotDoubleConnect(intersection, union, ind3set): # We make sure that there are two edges to every independent 3-set
                return False
        return True

    def J3(cone1, cone2, cone3): # Collapsing rule if three vertices in G form an independent j3
        union = cone1 | cone2 | cone3 # We look at the union of the neighborhoods of the cones
        for nonEdge in self.e2:
            if isNotConnect(union, nonEdge): # We make sure it connects to every non-edge
                return False
        return True

    def E3(cone1, cone2, cone3): # Collapsing rule if three vertices in G are not connected
        union = cone1 | cone2 | cone3 # We look at the union of the neighborhoods of the cones
        for edge in self.k2:
            if isNotConnect(union, edge): # We make sure it connects to every edge
                return False

        intersection = (cone1 & cone2) | (cone1 & cone3) | (cone2 & cone3) # We also need to look at the union of the pairwise intersections of the neighborhoods
        for nonEdge in self.e2:
            if isNotDoubleConnect2(intersection, union, nonEdge): # We make sure that there are two edges to every non-edge
                return False
        return True

    def J4(cone1, cone2, cone3, cone4): # Collapsing rule if four vertices in G form an independent j4
        union = cone1 | cone2 | cone3 | cone4 # We look at the union of the neighborhood of the cones
        return union == (1 << self.n)  - 1 # It has to include every vertex

    def E4(cone1, cone2, cone3, cone4): # Collapsing rule if four vertices in G are indepedents
        intersection = (cone1 & cone2) | (cone1 & cone3) | (cone1 & cone4) | (cone2 & cone3) | (cone2 & cone4) | (cone3 & cone4) # We look at the union of the pairwise intersections of the neighborhoods
        return union == (1 << self.n)  - 1 # It has to include every vertex


###################################################################################################################
# The Rest Is Unfinished
# Using these rules, we can make a general collapsing rule that takes as input the collapsed adjunct and parent and tests if they are compatible
def collapse(parentDict, adjunctDict, G, H):
    parentVertices = set(parentDict.keys())
    adjunctVertices = set(adjunctDict.keys())
    # First we check if the parent and the adjunct are compatible
    for vertex in parentVertices.intersection(adjunctVertices):
        if parentDict[vertex] != adjunctDict[vertex]:
            return False

    # We are only concerned with sub-structures involving the last vertex and the vertices between the adjunct vertex and the second to last vertex
    parentVertex = max(adjunctVertices)
    parentCone = adjunctDict[parentVertex]
    checkVertices = parentVertices.difference(adjunctVertices)

    # Find blue edges
    notNeighbors = set(notNeighboring(parentVertex, G)).intersection(checkVertices)
    for notNeighbor in notNeighbors:
        if not e2(parentCone, parentDict[notNeighbor], H):
            return False

    # Find edges:
    neighbors = set(neighboring(parentVertex, G)).intersection(checkVertices)
    for neighbor in neighbors:
        if not k2(parentCone, parentDict[neighbor], H):
            return False

    # Find independent 3-sets and first case of complement J3s:
    ind3set = []
    j3case1 = []
    unNeighbors = notNeighbors.copy()
    for i in range(len(notNeighbors) - 1):
        vertex = unNeighbors.pop()
        for unNeighbor in unNeighbors:
            if G[vertex][unNeighbor] == 0:
                ind3set.append({vertex, unNeighbor})
                if not e3(parentCone, parentDict[vertex], parentDict[unNeighbor], H):
                    return False

            elif G[vertex][unNeighbor] == 1:
                j3case1.append({vertex, unNeighbor})
                if not j3(parentCone, parentDict[vertex], parentDict[unNeighbor], H):
                    return False

    # Find second case of complement J3s:
    j3case2 = []
    for neighbor in neighbors:
        for notNeighbor in notNeighbors:
            if G[neighbor][notNeighbor] == 0:
                j3case2.append({neighbor, notNeighbor})
                if not j3(parentCone, parentDict[neighbor], parentDict[notNeighbor], H):
                    return False


    # Find independent 4-sets
    ind4set = []
    indLength = len(ind3set)
    for i in range(indLength - 2):
        for j in range(i + 1, indLength - 1):
            blue1 = ind3set[i]
            blue2 = ind3set[j]
            if blue1.intersection(blue2) != set():
                blue3 = blue1.symmetric_difference(blue2)
                if blue3 in ind3set[j + 1:]:
                    ind4 = blue1.union(blue2)
                    ind4set.append(ind4)

                    listInd4 = list(ind4)
                    if not e4(parentCone, listInd4[0], listInd4[1], listInd4[2], H):
                        ruleTime += time.time() - startRule
                        return False

    j4case1 = []
    # Find first case of complement J4s
    for edge in j3case1:
        for notNeighbor in notNeighbors.difference(edge):
            listEdge = list(edge)
            vertex1 = listEdge[0]
            vertex2 = listEdge[1]
            if G[notNeighbor][vertex1] == 0 and G[notNeighbor][vertex2] == 0:
                j4case1.append({notNeighbor, vertex1, vertex2})
                if not j4(parentCone, parentDict[notNeighbor], parentDict[vertex1], parentDict[vertex2], H):
                    return False

    # Find the second case of complement J4s
    j4case2 = []
    for neighbor in neighbors:
        for ind3 in ind3set:
            listInd3 = list(ind3)
            vertex1 = listInd3[0]
            vertex2 = listInd3[1]
            if G[neighbor][vertex1] == 0 and G[neighbor][vertex2] == 0:
                j4case2.append({neighbor, vertex1, vertex2})
                if not j4(parentCone, parentDict[neighbor], parentDict[vertex1], parentDict[vertex2], H):
                    return False
    return True

################################################################################
# We now need to make a double tree with our (K3, J5)
collapseTime = 0
class Node(object):
    adjunctSequence = [1, 1, 2, 2, 3, 3, 4]        # This will be our sequence for determining adjuncts
    treeDict = {i:[] for i in range(1, 8)}         # This will keep track of the each of the levels of the tree
    mainBranchesDict = {i:[] for i in range(1, 8)} # This will keep track of the main branches of the tree
    H = None

    def __init__(self, G, vertices, verticesSetDict, isMain, isCollapsed):
        self.G = G
        self.vertices        = vertices        # This is a list of labelled vertices
        self.verticesSetDict = verticesSetDict # For each of these vertices, we need to keep track of which vertices in H they're connected to
        self.level = len(vertices)                     # The number of vertices determines the node level on the tree
        self.depth = len(verticesSetDict[vertices[0]]) # Current number of possible sequences of feasible cones available at the node
        self.isMain      = isMain      # Is it on the main branches of the tree?
        self.isCollapsed = isCollapsed # Has the algorithm already calculated the possible sequences of feasible cones at this node? (originally set to False)

        self.loc = len(Node.treeDict[self.level])
        Node.treeDict[self.level].append(self)
        if self.isMain:
            Node.mainBranchesDict[self.level].append(self)

        # We calculate the parent and adjunct, but only if the graph has more than two vertices
        self.parent = None
        self.adjunct = None
        if self.level > 1:
            # Sees if the parent already has been created
            parentVertices = self.vertices[:-1]
            parentGraph =  [[(self.G)[row][col] if (row in parentVertices and col in parentVertices) else None for col in range(len(self.G)) ] for row in range(len(self.G))]
            for node in Node.treeDict[self.level - 1]:
                if node.G == parentGraph:
                    self.parent = node
                    if self.isMain:
                        if node not in Node.mainBranchesDict[self.level - 1]:
                            node.isMain = True
                            Node.mainBranchesDict[self.level - 1].append(node)
                    break
            else:
                # Creates a parent node
                parentDict = {vertex:[] for vertex in parentVertices}
                newNode = Node(parentGraph, parentVertices, parentDict, self.isMain, False)
                self.parent = newNode

            # Finds or creates an adjunct node
            adjunctVertices = self.vertices[0 : Node.adjunctSequence[self.level - 1] - 1] + [self.vertices[-1]]
            adjunctGraph = [[self.G[row][col] if (row in adjunctVertices and col in adjunctVertices) else None for col in range(len(self.G)) ] for row in range(len(self.G))]
            for node in Node.treeDict[Node.adjunctSequence[self.level - 1]]:
                if node.G == adjunctGraph:
                    self.adjunct = node
                    break

            else:
                adjunctDict = {vertex: [] for vertex in adjunctVertices}
                newNode = Node(adjunctGraph, adjunctVertices, adjunctDict, False, False)
                self.adjunct = newNode

    # Gives a sequence of feasible cones at a certain depth in the dictionary
    def index(self, i):
        return {vertex:self.verticesSetDict[vertex][i] for vertex in self.verticesSetDict}

    # Collapses a node based on its ajunct and its parent
    def collapseNode(self):
        global collapseTime
        # Recursively, we need the parent and the adjunct collapsed
        parent = self.parent
        adjunct = self.adjunct
        last = self.vertices[-1]
        verticesSetDict = self.verticesSetDict
        Ggraph = self.G
        Hgraph = Node.H

        if not parent.isCollapsed:
            parent.collapseNode()
        if not adjunct.isCollapsed:
            adjunct.collapseNode()

        for i in range(parent.depth):
            for j in range(adjunct.depth):
                newLayer = parent.index(i)
                indexedAdjunct = adjunct.index(j)

                startTime = time.time()
                canCollapse = collapse(indexedAdjunct, newLayer, Ggraph, Hgraph)
                collapseTime += time.time() - startTime
                if canCollapse:
                    newLayer[last] = indexedAdjunct[last]
                    for vertex in verticesSetDict:
                        verticesSetDict[vertex].append(newLayer[vertex])
                    self.verticesSetDict = verticesSetDict
                    self.depth += 1
        self.isCollapsed = True

        return self

    def __str__(self):
        return "Graph:\n" + formatGraph(self.G) + "Dict:\n" + formatDict(self.verticesSetDict) + "\n"

# run #########################################################################
programTime = time.time()

# R(K4,J6,10) to find feasible cones in
with open('k4k4e_10.g6', 'r') as file:
    k4j4 = file.read().splitlines()
k4j4 = [decodeG6(graph, True) for graph in k4j4]
#printGraphs(k4j4)

# Make double tree with these R(K3, J5, 7)
with open('k3k5e_07.g6', 'r') as file:
    k3j5 = file.read().splitlines()
k3j5 = [decodeG6(graph, True) for graph in k3j5]
# printGraphs(k3j5)
"""
blankVertices = list(range(7))
blankverticesSetDict = {vertex:[] for vertex in range(7)}
# print(formatDict(blankverticesSetDict))

startTree = time.time()
for graph in k3j5:
    Node(graph, blankVertices, blankverticesSetDict, True, False)
treeTime = time.time() - startTree

for level in Node.treeDict:
    print("----------------------------------------LEVEL:" + str(level) + "----------------------------------------")
    for node in Node.treeDict[level]:
        print(node)
"""
'''
Htime = 0
totalCollapseTime = 0
for H in k4j4[0:1]:
    startH = time.time()
    cones = feasibleCones(H)
    Htime += time.time() - startH

    print("\n".join(str(row) for row in H))
    length = len(cones)

    Node.H = H
    rootNodes = Node.treeDict[1]
    for root in rootNodes:
        root.depth = length
        root.verticesSetDict = {vertex:cones for vertex in root.vertices}
        root.isCollapsed = True

    startTotalCollapse = time.time()
    for level in range(2, len(k3j5[0]) + 1):
        for node in Node.mainBranchesDict[level]:
            node.collapseNode()
    totalCollapseTime += time.time() - startTotalCollapse

    for level in Node.treeDict:
        print(level)
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
        for node in Node.treeDict[level]:
            print(node)
            print("\n")

    for level in Node.treeDict:
        for node in Node.treeDict[level]:
            node.depth = 0
            node.verticesSetDict = {vertex:[] for vertex in node.vertices}
            node.isCollapsed = False
programTime = time.time() - programTime

print(programTime)
print(treeTime)
print(Htime)
print(totalCollapseTime)
print(collapseTime)
print(ruleTime)
print(helpRuleTime)
'''
