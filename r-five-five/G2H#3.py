###################################################################################################################
# Enumerating graphs in R(K4,J5,18) by adding edges between R(K3, J5, 7) and R(K4, J4, 10)
###################################################################################################################
import time
from itertools import permutations
import numpy as np
import pandas as pd

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

def feasibleCones(H, n):
    # Add triangle conditions
    triangles = []
    maxClique(0, (1 << n) - 1, 0, triangles, H, n)
    triangles = [triangle for triangle in triangles if bin(triangle).count('1') == 3] # Looks only at at maximal cliques of three

    ind3sets = []
    maxSet(0, (1 << n) - 1, 0, ind3sets, H, n)
    ind3sets = [set for set in ind3sets if bin(set).count('1') == 3]

    solutions = []
    for i in range(1 << n):
        for triangle in triangles:
            if triangle & ~i == 0:
                break
        else:
            for ind3set in ind3sets:
                if i & ind3set == 0:
                    break
            else:
                solutions.append([i])

    return np.array(solutions, dtype=np.uint16)
###################################################################################################################

###################################################################################################################
# Repertoire of clique and j subgraph finding functions.

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

def H1(G, n, x):
    solution = 0
    pointer = 1 << n
    for row in G:
        pointer >>= 1
        if row & x != 0:
            solution |= pointer
    return solution

def H2(G, n, x):
    solution = 0
    pointer = 1 << n
    for row in G:
        pointer >>= 1
        check = row & x
        while check & (check - 1) != 0:
            length = check.bit_length()
            if G[n - length] & row & x != 0:
                solution |= pointer
                break
            check ^= 1 << (length - 1)
    return solution

def H3(G, n, x):
    solution = 0
    pointer = 1 << n
    for row in G:
        pointer >>= 1
        check = row & x
        while check != 0:
            length = check.bit_length()
            vertex = 1 << (length - 1)
            neighborhood = G[n - length]
            if (x & neighborhood & ~row & ~pointer) != 0 or (x & row & ~neighborhood & ~vertex)!= 0:
                solution |= pointer
                break
            check ^= vertex
    return solution

class H(object):
    def __init__(self, h, n):
        self.n = n # Number of vertices
        self.h = h

        flip = (1 << n) - 1
        self.flip = flip

        self.h1Dict = np.array([H1(h, n, i) for i in range(flip + 1)], dtype=np.uint16)
        self.h1Dict2 = np.array([H1(h, n, i) for i in range(flip, -1, -1)], dtype=np.uint16)

        compH = complement(h, n)
        self.h1CompDict = np.array([H1(compH, n, i) for i in range(flip, -1, -1)], dtype=np.uint16)
        self.h2Dict = np.array([H2(compH, n, i) for i in range(flip, -1, -1)], dtype=np.uint16)
        self.h3Dict = np.array([H3(compH, n, i) for i in range(flip, -1, -1)], dtype=np.uint16)
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
    return list(permutations(range(n)))

def unPermuteTuple(perm, n): # Finds inverse of a permutation
    return [perm.index(i) for i in range(n)]

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
    nodesDict = {i:[] for i in range(1, 9)} # Keeps track of all nodes
    mainDict = {i:[] for i in range(1, 9)} # Keeps track of main nodes
    permsDict = {i:perms(i) for i in range(1, 9)} # Static dictionary, stores permutatations of tuples of different lengths
    adjunctDict = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 8: 5} # Static dictionary, stores the adjunct numbers for graphs of different sizes
    adjunctIndicesDict = {1: 1, 2: 1, 3: int('101', 2), 4: int('1001', 2), 5: int('11001', 2), 6: int('111001', 2), 7: int('1110001', 2), 8: int('11110001', 2)} # Static dictionary, keeps track of the indices of vertices in the adjunct

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
                self.parent = (Node(n - 1, parentGraph, set(), False, isMain), [i for i in range(n - 1)])
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
                self.adjunct = (Node(adjunctNum, adjunctGraph, set(), False, False), [i for i in range(adjunctNum)])
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

    def collapse(self, H):
        parent = self.parent
        adjunct = self.adjunct
        if parent[0].isCollapsed == False:
            parent[0].collapse(H)
        if adjunct[0].isCollapsed == False:
            adjunct[0].collapse(H)

        parentNeighborhoods = parent[0].neighborhoods[:, parent[1]]
        adjunctNeighborhoods = adjunct[0].neighborhoods[:, adjunct[1]]

        shared = adjunct[0].n - 1
        parentLength = self.n - 1

        keys = ["key" + str(i) for i in range(shared)]
        left = pd.DataFrame(parentNeighborhoods, columns=keys + ["column" + str(i) for i in range(shared, parentLength)])
        right = pd.DataFrame(adjunctNeighborhoods, columns=(keys + ["column" + str(parentLength)]))
        if shared == 0:
            left.insert(0, "temp", 0)
            right.insert(0, "temp", 0)
            posNeighborhoods = pd.merge(left, right, how="inner", on="temp", sort=False)
            posNeighborhoods.drop("temp", axis=1, inplace=True)

        else:
            posNeighborhoods = pd.merge(left, right, how="inner", on=keys, sort=False)

        posNeighborhoods = posNeighborhoods.to_numpy()

        boolNeighborhoods = np.full(len(posNeighborhoods), True)
        vectorized = np.transpose(posNeighborhoods)
        growthCone = posNeighborhoods[:, -1]

        for sub in self.e4:
            cone1, cone2, cone3 =  vectorized[sub[0]], vectorized[sub[1]], vectorized[sub[2]]
            intersection = (growthCone & cone1) | (growthCone & cone2) | (growthCone & cone3) | (cone1 & cone2) | (cone1 & cone3) | (cone2 & cone3)
            boolNeighborhoods &= intersection == H.flip

        for sub in self.j4:
            cone1, cone2, cone3 =  vectorized[sub[0]], vectorized[sub[1]], vectorized[sub[2]]
            union = growthCone | cone1 | cone2 | cone3
            boolNeighborhoods &= union == H.flip

        for sub in self.j3:
            cone1, cone2 =  vectorized[sub[0]], vectorized[sub[1]]
            union = growthCone | cone1 | cone2
            boolNeighborhoods &= H.h1CompDict[union] & ~union == 0

        for sub in self.e3:
            cone1, cone2 = vectorized[sub[0]], vectorized[sub[1]]
            union = growthCone | cone1 | cone2
            boolNeighborhoods &= H.h1Dict2[union] & ~union == 0
            intersection = (growthCone & cone1) | (growthCone & cone2) | (cone1 & cone2)
            boolNeighborhoods &= H.h1CompDict[union] & ~intersection == 0

        for sub in self.e2:
            union = growthCone | vectorized[sub]
            boolNeighborhoods &= H.h3Dict[union] & ~union == 0

        for sub in self.k2:
            intersection = growthCone & vectorized[sub]
            boolNeighborhoods &= H.h1Dict[intersection] & intersection == 0

        self.neighborhoods = posNeighborhoods[boolNeighborhoods]
        self.isCollapsed = True
        """
        if self.isMain:
            clearLine(8)
            Node.printCollapse()
        """
        return self
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
    Gs = [Node(gSize, g, None, False, True) for g in listG]

    for h in Hs:
        start = time.time()
        print("********************************************************************************************************************")
        #Node.printCollapse() # Showing the progress in collapsing the main nodes

        startingNeighborhoods = feasibleCones(h.h, hSize)
        for root in Node.nodesDict[1]: # Sets the root node for this H
            root.neighborhoods = startingNeighborhoods
            root.isCollapsed = True

        for i in range(2, gSize + 1): # Collapses nodes
            for node in Node.mainDict[i]:
                node.collapse(h)
        """
        for g in Gs: # Takes successful gluings, converts them into new adjacency matrices
            for neighborhood in g.neighborhoods:
                success.append(joinMatrix(g.graph, gSize, h.h, hSize, neighborhood))
                count += 1
        """
        for g in Gs:
            for neighborhood in g.neighborhoods:
                count += 1

        for i in range(1, gSize + 1): # Resets the nodes for the next H
            for node in Node.nodesDict[i]:
                node.neighborhoods = set()
                node.isCollapsed = False
        print("\nThere were " + str(count) + " successful gluings")
        print("\nCollapse took " + str(time.time() - start))
    return count

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

with open('RK4J6/dataset_k4kme/k4k4e_05.g6', 'r') as file:
    k4j4 = file.read().splitlines()
k4j4 = [formatGraph(decodeG6(graph)) for graph in k4j4] # Relevant H's

with open('RK4J6/dataset_k3kme/k3k5e_05.g6', 'r') as file:
    orig = file.read().splitlines()
k3j5 = [formatGraph(decodeG6(graph)) for graph in orig]  # Relevant G's

start = time.time()
gluings = glueG2H(k3j5, 5, k4j4, 5, Node) # Glues G's to H's
print("Total time: ", time.time() - start)
print("Total gluings: ", gluings)

"""
for glue in gluings:
    if not isk4j5(glue, 9):
        print("error")

compressedGluings = [compressG6(glue, 18) + "\n" for glue in gluings] # Compresses successful gluings
glueFile = open('glueFile2.txt', 'w')
glueFile.writelines(compressedGluings) # Writes gluings to a file
glueFile.close()
"""
