###################################################################################################################
# Enumerating graphs in R(K4,J5,18) by adding edges between R(K3, J5, 7) and R(K4, J4, 10)
###################################################################################################################
from ortools.sat.python import cp_model
import numpy as np
import time
import itertools

###################################################################################################################
# Returns convenient strings for printing

def printBit(decimal, n): # Binary form of decimal number
    return bin(int(decimal))[2:].zfill(n)

def printBitList(decimalList, n): # Binary form of tuple of decimal numbers
    return "\n".join(tuple(printBit(decimal, n) for decimal in decimalList))

def printDict(dict): # Key value pairs of a dictionary on seperate lines.
    return "\n".join(str(key) + ": " + str(dict[key]) for key in dict) + "\n"
###################################################################################################################

###################################################################################################################
# Decoding and Formating

def decodeG6(compressed): # Decodes graphs in g6 format to their adjacency matrixes
    n = ord(compressed[0]) - 63 # number of vertices
    length = int(n*(n - 1)/2) # number of 1's and 0's representing upper triangle of adjacency matrix

    bitVect = "" # Our bit vector will be represented by a string
    for character in compressed[1:]:
        bitVect += bin(ord(character) - 63)[2:].zfill(6) # subtract 63 from each byte and string the information together
    bitVect = bitVect[:length] # make sure our bit vector is the correct length

    adjacencyMatrix = [[0 for i in range(n)] for j in range(n)] # Our adjacency matrix will be a list of lists
    index = 0
    for column in range(1, n):
        for row in range(column):
            adjacencyMatrix[row][column] = adjacencyMatrix[column][row] = 1 ^ int(bitVect[index]) # We iterate through the rows and the columns, filling them out as we go
            index += 1
    return adjacencyMatrix

def formatBitList(bitVect): # Takes a list bit vector and converts it into an int representing the bit vector
    num = 0
    for i in bitVect[:-1]:
        num = (num + i) << 1
    return num + bitVect[-1]

def formatGraph(adjacencyMatrix):# Takes lists of lists adjacency matrix and converts it into a tuple of ints
    return tuple(formatBitList(row) for row in adjacencyMatrix)
###################################################################################################################

###################################################################################################################
# Finds feasible cones: subsets of the original graph that do not contain triangles and whose complements do not contain independent 3-sets.
# n will always denote the total number of vertices.

def bitIndex(bitVect, i): # Finds ith digit, starting from the right
    return (bitVect & (1 << i)) >> i

def expand(bitSet, n): # Expands a set represented in bits to its individual elements, also represented in bits
    return [1 << i for i in range(n) if bitIndex(bitSet, i)]

def maxClique(R, P, X, maxCliques, adjacencyMatrix, n): # Bron-Kerbosch algoithm for finding maximal cliques: https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    if P == 0 and X == 0:
        maxCliques.append(R)
        return None
    u = n - (P | X).bit_length()
    for v in expand(P & ~adjacencyMatrix[u], n):
        neighbors = adjacencyMatrix[n - v.bit_length()]
        maxClique(R | v, P & neighbors, X & neighbors, maxCliques, adjacencyMatrix, n)
        P = P & ~v
        X = X | v

def complement(adjacencyMatrix, n): # Takes complement of a formatted adjacency matrix
    flip = (1 << n) - 1
    return tuple(flip ^ adjacencyMatrix[row] ^ (1 << n - row - 1) for row in range(n))

def maxSet(R, P, X, maxSets, adjacencyMatrix, n): # Finds maximal independent sets
    compMatrix = complement(adjacencyMatrix, n) # Uses complement
    return maxClique(R, P, X, maxSets, compMatrix, n)

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback): # Solution printer to keep track of solutions
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_vector = set()
    def on_solution_callback(self):
        solution = 0
        for v in self.__variables:
            solution = solution << 1
            solution += self.Value(v)
        self.__solution_vector.add((solution,))
    def solution_vector(self):
        return self.__solution_vector

def feasibleCones(H, n): # Algorithm to find feasible cones
    vertices = range(n)
    model = cp_model.CpModel() # Create model
    variables = [model.NewBoolVar('v'+str(i)) for i in vertices] # Make variables for each of the vertices determining if they will be present in the feasible cone

    # Add triangle conditions
    triangles = []
    maxClique(0, (1 << n) - 1, 0, triangles, H, n)
    triangles = [triangle for triangle in triangles if bin(triangle).count('1') == 3] # Looks only at at maximal cliques of three

    for triangle in triangles:
        model.AddBoolOr([variables[vertex].Not() for vertex in vertices if bitIndex(triangle, n - vertex - 1)]) # Ensures that a feasible cone doesn't contain all three vertices

    # Add complement triangle conditions
    ind3sets = []
    maxSet(0, (1 << n) - 1, 0, ind3sets, H, n)
    ind3sets = [set for set in ind3sets if bin(set).count('1') == 3]
    for set in ind3sets:
        model.AddBoolOr([variables[vertex] for vertex in vertices if bitIndex(set, n - vertex - 1)]) # Ensures that a feasible cone is connected to the independent 3-set

    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(variables)
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model, solution_printer)
    bitSolutions = solution_printer.solution_vector()
    return bitSolutions
###################################################################################################################

###################################################################################################################
# Repertoire of clique and j subgraph finding functions.

def findEdges(adjacencyMatrix, n): # Finds list of edges and non-edges from non-formatted adjacency matrix
    edgeSet = []
    nonEdgeSet = []
    for row in range(n - 1):
        for column in range(row + 1, n):
            if adjacencyMatrix[row][column] == 1:
                edgeSet.append((1 << (n - row - 1)) | (1 << (n - column - 1))) # If the adjacency matrix value is 1, add an edge
            else:
                nonEdgeSet.append( (1 << (n - row - 1)) | (1 << (n  - column - 1)) ) # Otherwise, it is a non edge
    return edgeSet, nonEdgeSet


def findSmaller(maximalList, order, n): # Finds cliques of order one less than maximal
    smaller = []
    for maximal in maximalList:
        size = bin(maximal).count('1')
        if size == order - 1:
            smaller.append(maximal) # The subgraph is of correct size.
        elif size == order:
             for i in range(n):
                sub = maximal & ~(1 << i)  # Possible subgraph of maximal clique of correct order
                if bitIndex(maximal, i) == 1 and sub not in smaller: # Ensures there are no duplicates
                    smaller.append(sub)
    return smaller

def findJ(smaller, larger, order): # Finds J graphs, infering from a larger and smaller list of indpendent sets or cliques
    j = []
    length = len(smaller)
    if length > 1: # Ensures there are enough of the smaller subgraphs
        for k in range(length - 1):
            for l in range(k + 1, length): # Iterates through all pairs  of smaller subgraphs
                first, second = smaller[k], smaller[l]
                if bin(first & second).count('1') == order - 2: # Checks the intersection on the set of their vertices is of the correct order
                    combo = first | second
                    if combo not in larger: # Ensures that the union of their sets of vertices is not in the list of larger subgraphs
                        j.append(combo)
    return j

def maximalToMaximum(maximal, order): # Finds the maximum order graphs in a list of maximal subgraphs
    return [sub for sub in maximal if bin(sub).count('1') == order]
###################################################################################################################

###################################################################################################################
# Defines a class for the H graphs, which are in R(K4, J4, 10)

def isContains(bitVect1, bitVect2): # Determines if the set represented by bitVect2 is contained within the set represented by bitVect1
    return (bitVect2 & ~bitVect1) == 0

def isNotConnect(bitVect1, bitVect2): # Determines if the intersection of sets represented by bitVect1 and bitVect2 is empty
    return (bitVect1 & bitVect2) == 0

# Helper function to check if at least element of the third set is contained in the intersection of the sets or if two of its elements are contained in the union of the two sets
def isNotDoubleConnect(intersection, union, bitVect): # Determine
    check = union & bitVect # We see how many bits are set in this number
    return isNotConnect(intersection, bitVect) and check & (check - 1) == 0 # The last portion checks if check only contains one vertex

class H(object):
    def __init__(self, graph, n):
        self.n = n # Number of vertices
        self.k2, self.e2 = findEdges(graph, n) # Find edges and non-edges

        graph = formatGraph(graph)
        self.graph = graph

        k3 = []
        maxClique(0, (1 << n) - 1, 0, k3, graph, n)
        self.k3 = maximalToMaximum(k3, 3) # Finds triangles

        e3 = []
        maxSet(0, (1 << n) - 1, 0, e3, graph, n)
        self.e3 = maximalToMaximum(e3, 3) # Finds independent 3-sets

        self.j3 = findJ(self.e2, self.e3, 3) # Finds independent J3's

    def K2(self, cone1, cone2): # Collapsing rule if two vertices in G are connected
        intersection = cone1 & cone2 # Inspects intersection of the neighborhoods of the cones
        for edge in self.k2:
            if (edge & ~intersection) == 0: # isContains(intersection, edge): # Ensures that it contains no edges
                return False

        union = cone1 | cone2 # Inspects union of the neighborhoods of the cones
        for ind3set in self.e3:
            if (union & ind3set) == 0: # isNotConnect(union, ind3set): # Ensures that it connects to every indpendent 3-set
                return False
        return True

    def E2(self, cone1, cone2): # Collapsing rule if two vertices in G are not connected
        union = cone1 | cone2 # Inspects union of the neighborhoods of the cones
        for sub in self.j3:
            if (union & sub) == 0: # isNotConnect(union, sub): # Ensures that it connects to every independent j3
                return False

        intersection = cone1 & cone2 # Inspects intersection of the two neighborhoods
        for ind3set in self.e3:
            check = union & ind3set
            if (intersection & ind3set) == 0 and check & (check - 1) == 0: # isNotDoubleConnect(intersection, union, ind3set): # Ensures that there are two edges to every independent 3-set
                return False
        return True

    def J3(self, cone1, cone2, cone3): # Collapsing rule if three vertices in G form an independent j3
        union = cone1 | cone2 | cone3 # Inspects the union of the neighborhoods of the cones
        for nonEdge in self.e2:
            if (union & nonEdge) == 0: # isNotConnect(union, nonEdge): # Ensures that it connects to every non-edge
                return False
        return True

    def E3(self, cone1, cone2, cone3): # Collapsing rule if three vertices in G are not connected
        union = cone1 | cone2 | cone3 # Inspects the union of the neighborhoods of the cones
        for edge in self.k2:
            if (union & edge) == 0: # isNotConnect(union, edge): # Ensures that it connects to every edge
                return False

        intersection = (cone1 & cone2) | (cone1 & cone3) | (cone2 & cone3) # Inspects the union of the pairwise intersections of the neighborhoods
        for nonEdge in self.e2:
            check = union & nonEdge
            if (intersection & nonEdge) == 0 and check & (check - 1) == 0: # isNotDoubleConnect(intersection, union, nonEdge): # Ensures that there are two edges to every non-edge
                return False
        return True

    def J4(self, cone1, cone2, cone3, cone4): # Collapsing rule if four vertices in G form an independent j4
        union = cone1 | cone2 | cone3 | cone4 # Inspects the union of the neighborhood of the cones
        return union == (1 << self.n) - 1 # Ensures it includes every vertex

    def E4(self, cone1, cone2, cone3, cone4): # Collapsing rule if four vertices in G are indepedents
        intersection = (cone1 & cone2) | (cone1 & cone3) | (cone1 & cone4) | (cone2 & cone3) | (cone2 & cone4) | (cone3 & cone4) # Inspects the union of the pairwise intersections of the neighborhoods
        return intersection == (1 << self.n) - 1 # Ensures it include every vertex
###################################################################################################################

###################################################################################################################
# Defines a class for the G graphs, storing them in a double tree structure suitable for collapsing the G graphs

def findIndices(bitVector, n): # Turns a bit vector into a list of vertices
    indices = []
    pointer = 1
    for i in range(0, n):
        if bitVector & pointer != 0:
            indices.append(n - i - 1) # Iterates through the bit vector, checking if the value is equal to 1
        pointer <<= 1
    return indices

def perms(n): # Finds all permutations of tuples of length n
    return list(itertools.permutations(range(n)))

def unPermuteTuple(perm, n): # Finds inverse of a permutation
    return tuple(perm.index(i) for i in range(n))

def permuteTuple(perm, aTuple): # Permutes a tuple
    return tuple(aTuple[index] for index in perm)

def permuteBit(perm, bitVector, n): # Permutes a bit vector
    permuted = 0
    for i in range(n):
        index = perm[i]
        permuted |= ((1 << (n - index - 1)) & bitVector) >> (n - index - 1) << (n - i - 1) # Locate required bit and add it to our new vector
    return permuted

def permuteMatrix(perm, matrix, n): # Permutes formatted adjacency matrix
    return tuple(permuteBit(perm, matrix[i], n) for i in perm)

def isomorphismList(adjacencyMatrix, nodeList, n, permutations): # Finds an isomorphism between a graph and a graph within the lists of nodes
    labellings = {permuteMatrix(permutation, adjacencyMatrix, n): unPermuteTuple(permutation, n) for permutation in permutations}
    for node in nodeList:
        graph = node.graph
        if graph in labellings:
            return (node, labellings[graph]) # Finds isomorphic graph and the mapping itself
    return False

def excludedJ3(k2, e2, adjacencyMatrix, n): # Finds J3 that consist of the last vertex, a non-neighbor, and a neighbor
    j3 = []
    for neighbor in k2:
        neighborPos = (1 << (n - neighbor - 1))
        neighbors = adjacencyMatrix[neighbor]
        for notNeighbor in e2: # Iterates through all non-neighbors and neighbors
            notNeighborPos = (1 << (n - notNeighbor - 1))
            if neighbors & notNeighborPos == 0: # Ensures that the non-neighbor and the neighbor are unconnected
                j3.append(neighborPos | notNeighborPos | 1) # I
    return j3

def excludedJ4(excludedJ3, e3): # Finds J4 that consist of the last vertex, two non-neighbors, and a neighbor
    j4 = []
    length = len(excludedJ3)
    if length > 1: # Ensures there are enough of the excluded J3, which are subgraphs of excluded J4
        for k in range(length - 1):
            for l in range(k + 1, length): # Iterates through all pairs of the excluded J3
                first, second = excludedJ3[k], excludedJ3[l]
                combo = 1 | (first ^ second)
                if bin(combo).count('1') == 3 and combo in e3: # Ensures that the two excluded J3 form an excluded J4
                    j4.append(first | second)
    return j4

def findSmaller2(maximalList, order, n): # Adjusted findSmaller()
    smaller = []
    for maximal in maximalList:
        size = bin(maximal).count('1')
        if size == order - 1:
            smaller.append(maximal)
        elif size == order:
             for i in range(n):
                sub = maximal & ~(1 << i)
                if bitIndex(maximal, i) == 1 and sub not in smaller and sub & 1 == 1: # Ensures the subgraph included the last vertex
                    smaller.append(sub)
    return smaller

def clearLine(n): # Clears previous n lines on console, used for displaying progress.  Code from https://itnext.io/overwrite-previously-printed-lines-4218a.
    lineUp = '\033[1A'
    lineClear = '\x1b[2K'
    for i in range(n):
        print(lineUp, end=lineClear)

class Node(object):
    nodesDict = {i:[] for i in range(1, 8)} # Keeps track of all nodes
    mainDict = {i:[] for i in range(1, 8)} # Keeps track of main nodes
    permsDict = {i:perms(i) for i in range(1, 8)} # Static dictionary, stores permutatations of tuples of different lengths
    adjunctDict = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3} # Static dictionary, stores the adjunct numbers for graphs of different sizes
    adjunctIndicesDict = {1: 1, 2: 1, 3: int('001', 2), 4: int('0001', 2), 5: int('00001', 2), 6: int('100001', 2), 7: int('1100001', 2)} # Static dictionary, keeps track of the indices of vertices in the adjunct

    def __init__(self, n, graph, neighborhoods, isCollapsed, isMain):
        self.graph = graph # Formatted adjacency matrix of graph
        self.n = n # Size of graph
        self.neighborhoods =  neighborhoods # Set of possible neighborhoods in H, each index in the tuple corresponding to a vertex
        self.isCollapsed = isCollapsed
        self.isMain = isMain

        Node.nodesDict[n].append(self)
        if isMain:
            Node.mainDict[n].append(self)

        if n > 1:
            # Creates a parent node
            parentGraph = tuple((row & ~1) >> 1 for row in graph[:-1])
            parent = isomorphismList(parentGraph, Node.nodesDict[n - 1], n - 1, Node.permsDict[n - 1])
            if not parent:
                self.parent = (Node(n - 1, parentGraph, set(), False, isMain), tuple(i for i in range(n - 1)))
            else:
                if isMain:
                    parent[0].isMain = True
                    Node.mainDict[n - 1].append(parent[0])
                self.parent = parent

            # Creates an adjunct node
            adjunctNum = Node.adjunctDict[n]
            adjunctGraph = graph[:adjunctNum - 1] + (graph[-1],)
            flipAdjunct = (1 << (n - adjunctNum + 1)) - 1
            adjunctGraph = tuple(((row & ~flipAdjunct) >>  (n - adjunctNum)) | (row & 1) for row in adjunctGraph)
            adjunct = isomorphismList(adjunctGraph, Node.nodesDict[adjunctNum], adjunctNum, Node.permsDict[adjunctNum])
            if not adjunct:
                self.adjunct = (Node(adjunctNum, adjunctGraph, set(), False, False), tuple(i for i in range(adjunctNum)))
            else:
                self.adjunct = adjunct

            # Finds important subgraphs
            flipGraph = (1 << n) - 1
            neighbors = graph[-1]
            k2 = findIndices(neighbors, n)
            notNeighbors = (flipGraph & ~neighbors) ^ 1
            e2 = findIndices(notNeighbors, n)
            notEdges = [1 | notNeighbor for notNeighbor in expand(notNeighbors, n)]

            j3plus = excludedJ3(k2, e2, graph, n)

            maxSets = []
            maxSet(1, notNeighbors, 0, maxSets, graph, n)
            ind4sets = maximalToMaximum(maxSets, 4)
            ind3sets = findSmaller2(maxSets, 4, n)
            j3sets = findJ(notEdges, ind3sets, 3)
            j3sets += j3plus
            j4sets = findJ(ind3sets, ind4sets, 4)
            j4sets += excludedJ4(j3plus, ind3sets)

            # Stores subgraphs as indices of vertices
            self.k2 = [i for i in k2 if i > adjunctNum - 2]
            self.e2 = [i for i in e2 if i > adjunctNum - 2]

            adjunctIndices = Node.adjunctIndicesDict[n]
            self.j3 = [findIndices(sub ^ 1, n) for sub in j3sets if not isContains(adjunctIndices, sub)]
            self.e3 = [findIndices(sub ^ 1, n) for sub in ind3sets if not isContains(adjunctIndices, sub)]
            self.j4 = [findIndices(sub ^ 1, n) for sub in j4sets if not isContains(adjunctIndices, sub)]
            self.e4 = [findIndices(sub ^ 1, n) for sub in ind4sets if not isContains(adjunctIndices, sub)]

    @classmethod
    def printCollapse(cls): # Prints out collapsing progress of the main nodes
        print("\n".join(str(key) + ": " + " ".join('x' if node.isCollapsed else 'o' for node in cls.mainDict[key]) for key in cls.mainDict))

    def collapse(self, H, neighborhoods):
        k2, e2, j3, e3, Hn = H.k2, H.e2, H.j3, H.e3, H.n

        parent = self.parent
        adjunct = self.adjunct
        if parent[0].isCollapsed == False:
            parent[0].collapse(H, parent[0].neighborhoods)
        if adjunct[0].isCollapsed == False:
            adjunct[0].collapse(H, adjunct[0].neighborhoods)

        parentNeighborhoods = {tuple(neighborhood[index] for index in parent[1]) for neighborhood in parent[0].neighborhoods}
        adjunctNeighborhoods = {tuple(neighborhood[index] for index in adjunct[1]) for neighborhood in adjunct[0].neighborhoods}
        shared = adjunct[0].n - 1
        parentLength = self.n - 1

        for adjunctTemp in adjunctNeighborhoods: # We iterate through all possible parent adjunct combinations
            growthCone = adjunctTemp[-1]
            for parentTemp in parentNeighborhoods:
                if parentTemp[:shared] == adjunctTemp[:-1]:
                    for sub in self.e4:
                        cone2, cone3, cone4 = parentTemp[sub[0]], parentTemp[sub[1]], parentTemp[sub[2]]
                        if not (growthCone & cone2) | (growthCone & cone3) | (growthCone & cone4) | (cone2 & cone3) | (cone2 & cone4) | (cone3 & cone4) == (1 << Hn) - 1: # Ensures it include every vertex
                            break
                    else:
                        for sub in self.j4:
                            if not growthCone | parentTemp[sub[0]] | parentTemp[sub[1]] | parentTemp[sub[2]] == (1 << Hn) - 1: # Ensures it includes every vertex
                                break

                        else:
                            flag = False
                            for sub in self.j3:
                                union = growthCone | parentTemp[sub[0]] | parentTemp[sub[1]] # Inspects the union of the neighborhoods of the cones
                                for nonEdge in e2:
                                    if (union & nonEdge) == 0: # isNotConnect(union, nonEdge): # Ensures that it connects to every non-edge
                                        flag = True
                                        break
                                if flag:
                                    break

                            else:
                                for sub in self.e3:
                                    cone2, cone3 = parentTemp[sub[0]], parentTemp[sub[1]]
                                    union = growthCone | cone2 | cone3 # Inspects the union of the neighborhoods of the cones
                                    for edge in k2:
                                        if (union & edge) == 0: # isNotConnect(union, edge): # Ensures that it connects to every edge
                                            flag = True
                                            break

                                    if flag:
                                        break

                                    intersection = (growthCone & cone2) | (growthCone & cone3) | (cone2 & cone3) # Inspects the union of the pairwise intersections of the neighborhoods
                                    for nonEdge in e2:
                                        check = union & nonEdge
                                        if (intersection & nonEdge) == 0 and check & (check - 1) == 0: # isNotDoubleConnect(intersection, union, nonEdge): # Ensures that there are two edges to every non-edge
                                            flag = True
                                            break

                                    if flag:
                                        break
                                else:
                                    for sub in self.e2:
                                        cone2 = parentTemp[sub]
                                        union = growthCone | cone2 # Inspects union of the neighborhoods of the cones
                                        for sub2 in j3:
                                            if (union & sub2) == 0: # isNotConnect(union, sub): # Ensures that it connects to every independent j3
                                                flag = True
                                                break

                                        if flag:
                                            break

                                        intersection = growthCone & cone2 # Inspects intersection of the two neighborhoods
                                        for ind3set in e3:
                                            check = union & ind3set
                                            if (intersection & ind3set) == 0 and check & (check - 1) == 0: # isNotDoubleConnect(intersection, union, ind3set): # Ensures that there are two edges to every independent 3-set
                                                flag = True
                                                break
                                        if flag:
                                            break

                                    else:
                                        for sub in self.k2:
                                            cone2 = parentTemp[sub]
                                            intersection = growthCone & cone2 # Inspects intersection of the neighborhoods of the cones
                                            for edge in k2:
                                                if (edge & ~intersection) == 0: # isContains(intersection, edge): # Ensures that it contains no edges
                                                    flag = True
                                                    break

                                            if flag:
                                                break

                                            union = growthCone | cone2 # Inspects union of the neighborhoods of the cones
                                            for ind3set in e3:
                                                if (union & ind3set) == 0: # isNotConnect(union, ind3set): # Ensures that it connects to every indpendent 3-set
                                                    flag = True
                                                    break

                                            if flag:
                                                break

                                        else:
                                            neighborhoods.add(parentTemp + (adjunctTemp[-1],))

        self.isCollapsed = True

        if self.isMain:
            clearLine(7)
            Node.printCollapse()

        return neighborhoods
###################################################################################################################

###################################################################################################################
# Final gluing algorithm for G's and H's

def transposeVect(bitVect, n, shift): # Transposes bit vector, represented as a int, into a tuple, while also shifting the values in the tuple by a certain number of binary place values
    return tuple((((1 << (n - i - 1)) & bitVect) >> (n - i - 1)) << shift for i in range(n))

def transposeMatrix(matrix, m, n): # Transposes formatted matrix
    transposed = tuple(0 for i in range(n))
    for i in range(m):
        newColumn = transposeVect(matrix[i], n, m - i - 1)
        transposed = tuple(a + b for a, b in zip(transposed,newColumn))
    return transposed

def joinMatrix(G, gSize, H, hSize, neighborhood): # Joins two graphs G and H, with knowledge of connections in between them as given by neighborhood
    n = 1 + gSize + hSize

    row1 = ((1 << gSize) - 1) << hSize # First row of new matrix

    rowHeader = 1 << (n - 1)
    column1Top = (rowHeader for i in range(gSize))
    shiftG = tuple(row << hSize for row in G)
    top = tuple(a + b + c for a, b, c in zip(column1Top, shiftG, neighborhood)) # The rest of the top consists of G and its neighborhood in H

    transposedNeighborhood = tuple(row << hSize for row in transposeMatrix(neighborhood, gSize, hSize))
    bottom = tuple(a + b for a, b in zip(transposedNeighborhood, H)) # The bottom consists of H and its neighborhood in G

    return (row1,) + top + bottom

def isk4j5(graph, n): # Checks if successful gluings are in R(K4, J5)
    maxCliques = []
    maxClique(0, (1 << n) - 1, 0, maxCliques, graph, n)
    for max in maxCliques:
        if bin(max).count('1') >= 4: # Ensures there are no cliques of order greater than 3
            return False

    maxSets = []
    maxSet(0, (1 << n) - 1, 0, maxSets, graph, n)

    smaller = []
    for max in maxSets:
        vertexCount = bin(max).count('1')
        if vertexCount >= 5: # Ensures there are no independent sets of order greater than 4
            return False

        if vertexCount == 4:
            for small in smaller:
                if bin(max & small).count('1') == 3: # Ensures there are no independent J5's of order 5
                    return False
            smaller.append(max)
    return True

def glueG2H(listG, gSize, listH, hSize, Node): # Glues together a list of G's and H's
    success = []
    count = 0
    Hs = [H(h, hSize) for h in listH]
    Gs = [Node(gSize, g, set(), False, True) for g in listG]

    for h in Hs:
        start = time.time()
        count = 0
        print("********************************************************************************************************************")
        Node.printCollapse() # Showing the progress in collapsing the main nodes

        startingNeighborhoods = feasibleCones(h.graph, hSize)
        for root in Node.nodesDict[1]: # Sets the root node for this H
            root.neighborhoods = startingNeighborhoods
            root.isCollapsed = True

        for i in range(2, gSize + 1): # Collapses nodes
            for node in Node.mainDict[i]:
                node.collapse(h, node.neighborhoods)

        for g in Gs: # Takes successful gluings, converts them into new adjacency matrices
            for neighborhood in g.neighborhoods:
                success.append(joinMatrix(g.graph, gSize, h.graph, hSize, neighborhood))
                count += 1

        for i in range(1, gSize + 1): # Resets the nodes for the next H
            for node in Node.nodesDict[i]:
                node.neighborhoods = set()
                node.isCollapsed = False

        print("\nThere were " + str(count) + " successful gluings")
        print("\nCollapse took " + str(time.time() - start))
    return success


###################################################################################################################

# run #############################################################################################################
def compressG6(formatMatrix, n): # Compresses formatted adjacency matrix into g6 format for storage, inverse of decodeG6()
    bitVect = 0
    count = 0
    for i in range(n - 1):
        pointer = 1 << (n - i - 2)
        for j in range(i + 1):
            bitVect <<= 1
            bitVect += (formatMatrix[j] & pointer) >> (n - i - 2)
            count += 1

    size = int(n*(n - 1)/2)
    size6 = size % 6 + size
    bitVect <<= size6 - size
    stringVect = (bin(bitVect)[2:]).zfill(size6)
    code = [int(stringVect[6 * i: 6 * i + 6], 2) + 63 for i in range(int(size6 / 6))]
    return chr(n + 63) + "".join(chr(int(stringVect[6 * i: 6 * i + 6], 2) + 63) for i in range(int(size6 / 6)))

with open('k4k4e_10.g6', 'r') as file:
    k4j4 = file.read().splitlines()
k4j4 = [decodeG6(graph) for graph in k4j4] # Relevant H's

with open('k3k5e_07.g6', 'r') as file:
    orig = file.read().splitlines()
k3j5 = [formatGraph(decodeG6(graph)) for graph in orig]  # Relevant G's

gluings = glueG2H(k3j5, 7, k4j4, 10, Node) # Glues G's to H's
#compressedGluings = [compressG6(glue, 9) + "\n" for glue in gluings] # Compresses successful gluings
#glueFile = open('glueFile.txt', 'w')
#glueFile.writelines(compressedGluings) # Writes gluings to a file
#glueFile.close()
print(len(gluings))
