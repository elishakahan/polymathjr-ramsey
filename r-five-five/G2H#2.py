###################################################################################################################
# Enumerating graphs in R(K4,J5,18) by adding edges between R(K3, J5, 7) and R(K4, J4, 7)
###################################################################################################################
from ortools.sat.python import cp_model
import numpy as np
import time

###################################################################################################################
# Some printing functions (who might instead return convenient strings)

# Print binary form of decimal number
def printBit(decimal, n):
    return bin(int(decimal))[2:].zfill(n)

# Print binary form of list of decimal numbers
def printBitList(decimalList, n):
    return str([printBit(decimal, n) for decimal in decimalList])

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

def formatBitString(bitVect): # Takes string bit vector and converts it into an int representing the bit vector
    return int(bitVect, 2)

def formatBitList(bitVect): # Takes a list bit vector and converts it into an int representing the bit vector
    num = 0
    for i in bitVect[:-1]:
        num = (num + i) << 1
    return num + bitVect[-1]

def formatGraph(adjacencyMatrix):# Takes lists of lists adjacency matrix and converts it into a list of ints
    return [formatBitList(row) for row in adjacencyMatrix]
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
    if len(smaller) > 1: # We make sure smaller is big enough
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

# Helper function to see if a one set in bit form contains another
def isContains(bitVect1, bitVect2):
    return (bitVect1 & bitVect2) == bitVect2

# Helper function to see if the intersection of one set with another has at least one element
def isNotConnect(bitVect1, bitVect2):
    return (bitVect1 & bitVect2) == 0

# Helper function to check if at least element of the third set is contained in the intersection of the sets or if two of its elements are contained in the union of the two sets
def isNotDoubleConnect(intersection, union, bitVect):
    check = union & bitVect # We see how many bits are set in this number
    return isNotConnect(intersection, bitVect) and check & (check - 1) == 0 # The last portion checks if check only contains one vertex

class H(object):
    def __init__(self, graph, n):
        self.n = n # Number of vertices
        self.k2, self.e2 = findEdges(graph, n) # Find edges and non-edges

        graph = formatGraph(graph)
        k3 = []
        maxClique(0, (1 << n) - 1, 0, k3, graph, n)
        self.k3 = maximalToMaximum(k3, 3) # Find triangles
        e3 = []
        maxSet(0, (1 << n) - 1, 0, e3, graph, n)
        self.e3 = maximalToMaximum(e3, 3) # Find independent 3-sets
        self.j3 = findJ(self.e2, self.e3, 3) # Find independent J3's

    def K2(self, cone1, cone2): # Collapsing rule if two vertices in G are connected
        intersection = cone1 & cone2 # We look at the intersection of the neighborhoods of the cones
        for edge in self.k2:
            if isContains(intersection, edge): # We make sure that the it contains no edges
                return False

        union = cone1 | cone2 # We look at the union of the neighborhoods of the cones
        for ind3set in self.e3:
            if isNotConnect(union, ind3set): # We make sure it connects to every indpendent 3-set
                return False
        return True

    def E2(self, cone1, cone2): # Collapsing rule if two vertices in G are not connected
        union = cone1 | cone2 # We look at the union of the neighborhoods of the cones
        for sub in self.j3:
            if isNotConnect(union, sub): # We make sure it connects to every independent j3
                return False

        intersection = cone1 & cone2 # We also need to look at the intersection of the two neighborhoods
        for ind3set in self.e3:
            if isNotDoubleConnect(intersection, union, ind3set): # We make sure that there are two edges to every independent 3-set
                return False
        return True

    def J3(self, cone1, cone2, cone3): # Collapsing rule if three vertices in G form an independent j3
        union = cone1 | cone2 | cone3 # We look at the union of the neighborhoods of the cones
        for nonEdge in self.e2:
            if isNotConnect(union, nonEdge): # We make sure it connects to every non-edge
                return False
        return True

    def E3(self, cone1, cone2, cone3): # Collapsing rule if three vertices in G are not connected
        union = cone1 | cone2 | cone3 # We look at the union of the neighborhoods of the cones
        for edge in self.k2:
            if isNotConnect(union, edge): # We make sure it connects to every edge
                return False

        intersection = (cone1 & cone2) | (cone1 & cone3) | (cone2 & cone3) # We also need to look at the union of the pairwise intersections of the neighborhoods
        for nonEdge in self.e2:
            if isNotDoubleConnect(intersection, union, nonEdge): # We make sure that there are two edges to every non-edge
                return False
        return True

    def J4(self, cone1, cone2, cone3, cone4): # Collapsing rule if four vertices in G form an independent j4
        union = cone1 | cone2 | cone3 | cone4 # We look at the union of the neighborhood of the cones
        return union == (1 << self.n)  - 1 # It has to include every vertex

    def E4(self, cone1, cone2, cone3, cone4): # Collapsing rule if four vertices in G are indepedents
        intersection = (cone1 & cone2) | (cone1 & cone3) | (cone1 & cone4) | (cone2 & cone3) | (cone2 & cone4) | (cone3 & cone4) # We look at the union of the pairwise intersections of the neighborhoods
        return intersection == (1 << self.n)  - 1 # It has to include every vertex

    def __str__(self):
        return "K2: " + printBitList(self.k2, 7) + "\n" + "K3: " + printBitList(self.k3, 7) + "\n" +\
               "E2: " + printBitList(self.e2, 7) + "\n" + "E3: " + printBitList(self.e3, 7) + "\n" +\
               "J3: " + printBitList(self.j3, 7)

graph = [[0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 1, 0, 0, 1],
         [1, 0, 1, 0, 1, 1, 1],
         [1, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 1, 0, 1],
         [0, 0, 1, 1, 0, 1, 0]]
###################################################################################################################

###################################################################################################################
# We begin the creation of our final double tree.
# We need to recursively determine the adjunct and parent of each of our graphs in R(K3, J5, 7)
# At the root nodes, which consist only of one vertex, we assign possible neighborhoods as the set of feasible cones
# We use recursively use the information from the adjunct and the parent of a node to collapse that node

# Turn a bit vector into a a list of vertices
def findIndices(bitVector, n):
    indices = []
    pointer = 1
    for i in range(0, n):
        if bitVector & pointer != 0:
            indices.append(n - i - 1) # We iterate through the bit vector, checking if the value is equal to 1
        pointer <<= 1
    return indices

# Do the same thing, but for every item in a list
def findIndicesList(collection, n):
    return [findIndices(sub, n) for sub in collection]

class collapsableNode(Node):
    def __init__(self, n, graph, vertices, neighborhoods, isCollapsed):
        super(Node, self, n, graph, vertices).__init__()

        adjunctComp =  ((1 << (self.n)) - 1) ^ (self.adjunctVertices)
        neighbors = adjunctComp & self.graph[-1]
        notNeighbors = adjunctComp & ~self.graph[-1]
        notEdges = [1 | notNeighbor for notNeighbor in expand(notNeighbors, self.n)]

        maxSets = []
        maxSet(1, neighbors, notNeighbors, maxSets, graph, self.n)
        ind4sets = maximalToMaximum(e4, 4) # Find independent 4-sets
        ind3sets = findSmaller(maxSets, 4, self.n) # Find independent 3-sets
        j3sets = findJ(notEdges, ind3sets, 3) # Find independent J3
        j4sets = findJ(ind3sets, ind4sets, 4) # Find independent J3

        self.k2 = findIndices(neighbors, self.n)
        self.e2 = findIndices(notNeighbors, self.n)
        self.j3 = [findIndices(sub ^ 1, self.n) for sub in j3sets]
        self.e3 = [findIndices(sub ^ 1, self.n) for sub in ind3sets]
        self.j4 = [findIndices(sub ^ 1, self.n) for sub in j4sets]
        self.e4 = [findIndices(sub ^ 1, self.n) for sub in ind4sets]

    def collapseNode(self, H):
        parentNeighborhoods = (self.parent).neighborhoods
        adjunctNeighborhoods = (self.adjunct).neighborhoods

        shared = (self.adjunct).n - 1 # Find shared portion of parent and adjunct to make sure they are equivalent
        K2 = findIndicesList(self.k2, self.n)
        J3 = findIndicesList(self.j3, self.n)
        J4 = findIndicesList(self.j4, self.n)
        E2 = findIndicesList(self.e2, self.n)
        E3 = findIndicesList(self.e3, self.n)
        E4 = findIndicesList(self.e4, self.n)
        for possibleParent in parentNeighborhoods:
            for possibleAdjunct in adjunctNeighborhoods: # We iterate through all possible parent adjunct combinations
                cone1 = possibleAdjunct[-1]
                flag = False
                if possibleParent[:shared] == possibleAdjunct[:-1]:
                    for sub in K2:
                        if not H.K2(cone1, possibleParent[sub[0]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in E2:
                        if not H.E2(cone1, possibleParent[sub[0]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in J3:
                        if not H.J3(cone1, possibleParent[sub[0]], possibleParent[sub[1]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in E3:
                        if not H.E3(cone1, possibleParent[sub[0]], possibleParent[sub[1]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in J4:
                        if not H.J4(cone1, possibleParent[sub[0]], possibleParent[sub[1]], possibleParent[sub[2]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in E4:
                        if not H.E4(cone1, possibleParent[sub[0]], possibleParent[sub[1]], possibleParent[sub[2]]):
                            flag = True
                            break
                    if flag == True:
                        continue
                    (self.neighborhoods).append(possibleParent + list(possibleAdjunct[-1]))

        self.isCollapsed = True
        return self
'''
class Node(object):
    roots = []
    tree = []

    def __init__(self, n, graph, vertices, neighborhoods, isCollapsed):
        (Node.tree).append(self)
        self.graph = graph # The adjacency matrix the node represents, in formatted form
        self.vertices = vertices  # Vertices of the original graph, stored in a list
        self.n = n # Number of vertices in the graph
        self.neighborhoods = neighborhoods # List of lists, where each list in the outer list is a possible sequence of neighborhoods for each vertex
        self.isCollapsed = isCollapsed # The node has been collapsed, all possibilies for neighborhoods for its vertices calculated

        # We calculate the parent and adjunct, but only if the graph has more than two vertices
        self.parent = None
        self.adjunct = None

        if n > 1:
            # Creates a parent node
            parentVertices = vertices[:-1]
            parentGraph = [(row & ~1) >> 1 for row in graph[:-1]]
            newNode = Node(n - 1, parentGraph, parentVertices, [], False)
            self.parent = newNode

            # Creates an adjunct node
            adjunctSequence = [1, 1, 2, 2, 3, 3, 4] # This sequence will determine the adjunct
            adjunctNum = adjunctSequence[n - 1] # We access the class instance adjunctSequence to determine the the number of vertices in the adjunct
            adjunctVertices = vertices[0 : adjunctNum - 1] + [vertices[-1]] # The definition of an adjunct of a graph
            adjunctIndices = ((1 << (adjunctNum - 1)) << (n - adjunctNum + 1)) | 1
            adjunctGraph = graph[:adjunctNum - 1] + [graph[-1]]
            adjunctGraph = [((row & adjunctIndices) >> (n - adjunctNum)) | (row & 1) for row in adjunctGraph]
            newNode = Node(adjunctNum, adjunctGraph, adjunctVertices, [], False)
            self.adjunct = newNode

            neighbors = graph[-1] & ~(adjunctIndices)
            nonNeighbors = (1 << n - 1) & ~graph[-1] & ~(adjunctIndices)
            self.k2, self.e2 = [1 | neighbor for neighbor in expand(neighbors, n)], [1 | nonNeighbor for nonNeighbor in expand(nonNeighbors, n)]

            e4 = []
            maxSet(1, nonNeighbors, neighbors, e4, graph, self.n)
            self.e4 = maximalToMaximum(e4, 3) # Find independent 4-sets
            self.e3 = findSmaller(e4, 4, self.n) # Find independent 3-sets
            self.j3 = findJ(self.e2, self.e3, 3) # Find independent J3
            self.j4 = findJ(self.e3, self.e4, 4) # Find independent J3

        else:
            (Node.roots).append(self)

    # Collapses a node based on its adjunct and its parent
    def collapseNode(self, H):
        # Recursively, we need the parent and the adjunct collapsed
        if not (self.parent).isCollapsed:
            (self.parent).collapseNode(H)
        if not (self.adjunct).isCollapsed:
            (self.adjunct).collapseNode(H)

        parentNeighborhoods = (self.parent).neighborhoods
        adjunctNeighborhoods = (self.adjunct).neighborhoods

        shared = (self.adjunct).n - 1 # Find shared portion of parent and adjunct to make sure they are equivalent
        K2 = findIndicesList(self.k2, self.n)
        J3 = findIndicesList(self.j3, self.n)
        J4 = findIndicesList(self.j4, self.n)
        E2 = findIndicesList(self.e2, self.n)
        E3 = findIndicesList(self.e3, self.n)
        E4 = findIndicesList(self.e4, self.n)
        for possibleParent in parentNeighborhoods:
            for possibleAdjunct in adjunctNeighborhoods: # We iterate through all possible parent adjunct combinations
                cone1 = possibleAdjunct[-1]
                flag = False
                if possibleParent[:shared] == possibleAdjunct[:-1]:
                    for sub in K2:
                        if not H.K2(cone1, possibleParent[sub[0]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in E2:
                        if not H.E2(cone1, possibleParent[sub[0]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in J3:
                        if not H.J3(cone1, possibleParent[sub[0]], possibleParent[sub[1]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in E3:
                        if not H.E3(cone1, possibleParent[sub[0]], possibleParent[sub[1]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in J4:
                        if not H.J4(cone1, possibleParent[sub[0]], possibleParent[sub[1]], possibleParent[sub[2]]):
                            flag = True
                            break
                    if flag == True:
                        continue

                    for sub in E4:
                        if not H.E4(cone1, possibleParent[sub[0]], possibleParent[sub[1]], possibleParent[sub[2]]):
                            flag = True
                            break
                    if flag == True:
                        continue
                    (self.neighborhoods).append(possibleParent + list(possibleAdjunct[-1]))

        self.isCollapsed = True
        return self
###################################################################################################################

# run #############################################################################################################
# R(K4,J6,10) to find feasible cones in
with open('k4k4e_10.g6', 'r') as file:
    k4j4 = file.read().splitlines()
k4j4 = [decodeG6(graph, True) for graph in k4j4]
#printGraphs(k4j4)
Hs = [H(graph, 10) for graph in k4j4]

# Make double tree with these R(K3, J5, 7)
with open('k3k5e_07.g6', 'r') as file:
    k3j5 = file.read().splitlines()
k3j5 = [decodeG6(graph, True) for graph in k3j5]
# printGraphs(k3j5)
Gs = [Node(7, formatAdjacencyMatrix(graph), list(range(7)), [], False) for graph in k3j5]

for h in Hs:
    neighborhoods = [[neighborhood] for neighborhood in h.feasibleCones]
    for root in Node.roots:
        root.neighborhoods = neighborhoods
        root.isCollapsed = True

    for g in Gs:
        g.collapseNode(h)

    for graph in Node.tree:
        graph.neighborhoods = []
'''
