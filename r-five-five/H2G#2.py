###################################################################################################################
# Enumerating graphs in R(K4,J5,18) by adding edges between R(K3, J5, 7) and R(K4, J4, 10)
###################################################################################################################
import time
from itertools import permutations

###################################################################################################################
# Returns convenient strings for printing

def printBit(decimal, n): # Binary form of decimal number
    return bin(int(decimal))[2:].zfill(n)

def printBitList(decimalList, n): # Binary form of tuple of decimal numbers
    return "\n".join(tuple(printBit(decimal, n) for decimal in decimalList))

def printDict(dict, n): # Key value pairs of a dictionary on seperate lines.
    return "\n".join(printBit(key, n) + ": " + printBit(dict[key], n) for key in dict) + "\n"
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
# Repertoire of clique and j subgraph finding functions.
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

def findEdges(graph, n): # Finds list of edges and non-edges from formatted adjacency matrix
    edgeSet = []
    nonEdgeSet = []
    for i in range(n - 1):
        iVertex = 1 << (n - i - 1)
        row = graph[i]
        for j in range(n - i - 1):
            jVertex = 1 << j
            pair = iVertex | jVertex
            if jVertex & row != 0:
                edgeSet.append(pair)
            else:
                nonEdgeSet.append(pair)
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
# Finds feasible cones: subsets of the vertices of graph whose complement subset do not contain independent J4 or independent 4-sets
# Places the feasible cones into intervals

def inside(bottom, top, n):
    insert = expand(bottom | ~top, n)
    cones = set()
    for i in range(1 << (n - len(insert))):
        cone = i
        for vertex in insert:
            index = vertex.bit_length()
            cone = (cone >> (index - 1) << index) | (cone & (vertex - 1))
        cones.add(cone | bottom)
    return cones

def minFeasibleCones(G, n):
    # Find independent 4-sets
    maxSets = []
    maxSet(0, (1 << n) - 1, 0, maxSets, G, n)
    ind4sets = maximalToMaximum(maxSets, 4) # Looks only at at maximal cliques of three

    # Find independent J4
    ind3sets = findSmaller(maxSets, 4, n)
    j4sets = findJ(ind3sets, ind4sets, 4)

    cones = set()

    possible = set(range(1 << n))
    sorted = list(range(1<<n))
    sorted.sort(key=lambda x : bin(x).count('1'))

    top = (1 << n) - 1
    for i in sorted:
        if i in possible:
            for ind4set in ind4sets:
                check = i & ind4set
                if check & (check - 1) == 0:
                    possible.remove(i)
                    break
            else:
                for j4set in j4sets:
                    if i & j4set == 0:
                        possible.remove(i)
                        break
                else:
                    supersets = inside(i, top, n)

                    possible.difference_update(supersets)
                    possible.add(i)

                    cones.update(supersets)
    return possible, cones.difference(possible)

def findTop(tops, bottoms, newBottom, n):
    posTops = [0]
    newBottomVertices = {1 << i for i in range(n) if bitIndex(newBottom, i)}
    for top, bottom in zip(tops, bottoms):
        nextPos = []
        if newBottom & ~top == 0:
            bottomVertices = expand(bottom, n)
            for pos in posTops:
                if pos & bottom == 0:
                    for vertex in bottomVertices:
                        if vertex not in newBottomVertices:
                            nextPos.append(pos | vertex)
                else:
                    nextPos.append(pos)
            posTops = nextPos
    posTops.sort(key=lambda x : bin(x).count('1'))
    return posTops[0]

def intervals(G, n):
    minCones, cones = minFeasibleCones(G, n)
    orderMin, orderCone = list(minCones), list(cones)
    bitSort = lambda x : bin(x).count('1')
    orderMin.sort(key=bitSort), orderCone.sort(key=bitSort)
    ordered = orderMin + orderCone

    cones.update(minCones)
    bottoms, tops = [], []

    flip = (1 << n) - 1
    while cones != set():
        bottom = ordered[0]
        ordered = ordered[1:]
        if bottom in cones:
            top = flip & ~findTop(tops, bottoms, bottom, n)
            tops.append(top)
            bottoms.append(bottom)
            cones.difference_update(inside(bottom, top, n))

    return bottoms, tops
###################################################################################################################

###################################################################################################################
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

class G(object):
    def __init__(self, g, n):
        self.graph = g
        self.n = n

        flip = (1 << n) - 1
        self.flip = flip

        self.h1Dict = {i: H1(g, n, i) for i in range(1 << n)}

        compG = complement(g, n)
        self.h1CompDict = {flip & ~i: H1(compG, n, i) for i in range(flip + 1)}
        self.h2Dict = {flip & ~i: H2(compG, n, i) for i in range(flip + 1)}
        self.h3Dict = {flip & ~i: H3(compG, n, i) for i in range(flip + 1)}

def K2_1(tops, bottom1, bottom2, vertex1, vertex2, h1Dict):
    intersection = bottom1 & bottom2
    h1 = h1Dict[intersection]
    if intersection & h1 != 0:
        return False
    tops[vertex1] &= ~(h1 & bottom2)
    tops[vertex2] &= ~(h1 & bottom1)
    return True

def K2_2(bottoms, top1, top2, vertex1, vertex2, h2Dict):
    union = top1 | top2
    h2 = h2Dict[union]
    if h2 & ~union != 0:
        return False
    bottoms[vertex1] |= h2 & ~top2
    bottoms[vertex2] |= h2 & ~top1
    return True

def I2(bottoms, top1, top2, vertex1, vertex2, h2Dict, h3Dict):
    union = top1 | top2
    h3 = h3Dict[union]
    if h3 & ~union != 0:
        return False
    bottoms[vertex1] |= h3 & ~top2
    bottoms[vertex2] |= h3 & ~top1

    intersection = top1 & top2
    h2 = h2Dict[union]
    if h2 & ~intersection != 0:
        return False
    bottoms[vertex1] |= h2
    bottoms[vertex2] |= h2
    return True

def K3(tops, bottom1, bottom2, bottom3, vertex1, vertex2, vertex3):
    if bottom1 & bottom2 & bottom3 != 0:
        return False
    tops[vertex1] &= ~(bottom2 & bottom3)
    tops[vertex2] &= ~(bottom1 & bottom3)
    tops[vertex3] &= ~(bottom1 & bottom2)
    return True

def J3(bottoms, top1, top2, top3, vertex1, vertex2, vertex3, h1CompDict):
    union = top1 | top2 | top3
    h1Comp = h1CompDict[union]
    if h1Comp & ~union != 0:
        return False
    bottoms[vertex1] |= h1Comp & ~(top2 | top3)
    bottoms[vertex2] |= h1Comp & ~(top1 | top3)
    bottoms[vertex3] |= h1Comp & ~(top1 | top2)

    return True

def I3(bottoms, top1, top2, top3, vertex1, vertex2, vertex3, flip):
    if top1 | top2 | top3 != flip:
        return False
    bottoms[vertex1] |= flip & ~(top2 | top3)
    bottoms[vertex2] |= flip & ~(top1 | top3)
    bottoms[vertex3] |= flip & ~(top1 | top2)
    return True
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
    return tuple(perm.index(i) for i in range(n))

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

def clearLine(n): # Clears previous n lines on console, used for displaying progress.  Code from https://itnext.io/overwrite-previously-printed-lines-4218a.
    lineUp = '\033[1A'
    lineClear = '\x1b[2K'
    for i in range(n):
        print(lineUp, end=lineClear)

class Node(object):
    nodesDict = {i:[] for i in range(1, 9)} # Keeps track of all nodes
    mainDict = {i:[] for i in range(1, 9)} # Keeps track of main nodes
    permsDict = {i:perms(i) for i in range(1, 9)} # Static dictionary, stores permutatations of tuples of different lengths
    adjunctDict = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4} # Static dictionary, stores the adjunct numbers for graphs of different sizes

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
                self.parent = (Node(n - 1, parentGraph, [], False, isMain), tuple(i for i in range(n - 1)))
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
                self.adjunct = (Node(adjunctNum, adjunctGraph, [], False, False), tuple(i for i in range(adjunctNum)))
            else:
                self.adjunct = adjunct

            # Finds important subgraphs
            k2, i2 = findEdges(graph, n) # Find edges and non-edges

            k3 = []
            maxClique(0, (1 << n) - 1, 0, k3, graph, n)
            k3 = maximalToMaximum(k3, 3) # Finds triangles

            i3 = []
            maxSet(0, (1 << n) - 1, 0, i3, graph, n)
            i3 = maximalToMaximum(i3, 3) # Finds independent 3-sets

            j3 = findJ(i2, i3, 3) # Finds independent J3's

            self.k2 = [findIndices(sub, n) for sub in k2]
            self.i2 = [findIndices(sub, n) for sub in i2]
            self.k3 = [findIndices(sub, n) for sub in k3]
            self.j3 = [findIndices(sub, n) for sub in j3]
            self.i3 = [findIndices(sub, n) for sub in i3]

    @classmethod
    def printCollapse(cls): # Prints out collapsing progress of the main nodes
        print("\n".join(str(key) + ": " + " ".join('x' if node.isCollapsed else 'o' for node in cls.nodesDict[key]) for key in cls.nodesDict))

    def collapseSequence(self, bottoms, tops, G):
        for bottom, top in zip(bottoms, tops):
            if bottom & ~top != 0:
                return False

        originalBottoms = originalTops = -1
        while originalBottoms != bottoms or originalTops != tops:
            originalBottoms, originalTops = bottoms[:], tops[:]
            for nonEdge in self.i2:
                if not I2(bottoms, tops[nonEdge[0]], tops[nonEdge[1]], nonEdge[0], nonEdge[1], G.h2Dict, G.h3Dict):
                    return False
            for edge in self.k2:
                if not K2_1(tops, bottoms[edge[0]], bottoms[edge[1]], edge[0], edge[1], G.h1Dict):
                    return False
                if not K2_2(bottoms, tops[edge[0]], tops[edge[1]], edge[0], edge[1], G.h2Dict):
                    return False
            for ind3set in self.i3:
                if not I3(bottoms, tops[ind3set[0]], tops[ind3set[1]], tops[ind3set[2]], ind3set[0], ind3set[1], ind3set[2], G.flip):
                    return False
            for triangle in self.k3:
                if not K3(tops, bottoms[triangle[0]], bottoms[triangle[1]], bottoms[triangle[2]], triangle[0], triangle[1], triangle[2]):
                    return False
            for j3set in self.j3:
                if not J3(bottoms, tops[j3set[0]], tops[j3set[1]], tops[j3set[2]], j3set[0], j3set[1], j3set[2], G.h1CompDict):
                    return False
        return True

    def collapseNode(self, G):
        clearLine(9)
        Node.printCollapse()
        parent = self.parent
        adjunct = self.adjunct

        if parent[0].isCollapsed == False:
            parent[0].collapseNode(G)
        if adjunct[0].isCollapsed == False:
            adjunct[0].collapseNode(G)

        parentNeighborhoods = [([neighborhood[0][index] for index in parent[1]], [neighborhood[1][index] for index in parent[1]]) for neighborhood in parent[0].neighborhoods]
        adjunctNeighborhoods = [([neighborhood[0][index] for index in adjunct[1]], [neighborhood[1][index] for index in adjunct[1]]) for neighborhood in adjunct[0].neighborhoods]

        shared = adjunct[0].n - 1
        neighborhoods = []
        for parentNeighborhood in parentNeighborhoods:
            middleBottoms = parentNeighborhood[0][shared:]
            middleTops = parentNeighborhood[1][shared:]
            for adjunctNeighborhood in adjunctNeighborhoods:
                beginBottoms = [parentNeighborhood[0][i] | adjunctNeighborhood[0][i] for i in range(shared)]
                beginTops = [parentNeighborhood[1][i] & adjunctNeighborhood[1][i] for i in range(shared)]

                bottoms = beginBottoms + middleBottoms + adjunctNeighborhood[0][-1:]
                tops = beginTops + middleTops + adjunctNeighborhood[1][-1:]
                if self.collapseSequence(bottoms, tops, G):
                    neighborhoods.append((bottoms, tops))

        self.neighborhoods = neighborhoods
        self.isCollapsed = True
        return self

    def sequenceSplit(self, bottoms, tops, G):
        split = []
        for i in range(self.n):
            bottom, top = bottoms[i], tops[i]
            if bottom != top:
                vertex = 1 << ((top & ~bottom).bit_length() - 1)
                include = (bottoms[:i] + [bottom | vertex] + bottoms[i + 1:], tops[:])
                if self.collapseSequence(include[0], include[1], G):
                    split.append(include)
                notInclude = (bottoms[:], tops[:i] + [top & ~vertex] + tops[i + 1:])
                if self.collapseSequence(notInclude[0], notInclude[1], G):
                    split.append(notInclude)
                return split

    def collapseH(self, G):
        coneNeighborhoods = []
        neighborhoods = self.neighborhoods
        while neighborhoods != []:
            bottoms, tops = neighborhoods.pop()
            if bottoms ==  tops:
                coneNeighborhoods.append(tuple(bottoms))
            else:
                neighborhoods.extend(self.sequenceSplit(bottoms, tops, G))
        return coneNeighborhoods


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
    transposedNeighborhood = transposeMatrix(neighborhood, hSize, gSize)
    top = tuple(a + b + c for a, b, c in zip(column1Top, shiftG, transposedNeighborhood)) # The rest of the top consists of G and its neighborhood in H

    shiftedNeighborhood = tuple(row << hSize for row in neighborhood)
    bottom = tuple(a + b for a, b in zip(shiftedNeighborhood, H)) # The bottom consists of H and its neighborhood in G

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
    Gs = [G(g, gSize) for g in listG]
    Hs = [Node(hSize, h, [], False, True) for h in listH]

    for g in Gs:
        start = time.time()
        count = 0
        print("********************************************************************************************************************")
        Node.printCollapse() # Showing the progress in collapsing the main nodes

        bottoms, tops = intervals(g.graph, gSize)
        startingIntervals = [([bottom], [top]) for bottom, top in zip(bottoms, tops)]
        for root in Node.nodesDict[1]: # Sets the root node for this H
            root.neighborhoods = startingIntervals
            root.isCollapsed = True

        for i in range(2, hSize + 1): # Collapses nodes
            for node in Node.mainDict[i]:
                node.collapseNode(g)

        for h in Hs: # Takes successful gluings, converts them into new adjacency matrices
            coneNeighborhoods = h.collapseH(g)
            for neighborhood in coneNeighborhoods:
                success.append(joinMatrix(g.graph, gSize, h.graph, hSize, neighborhood))
                count += 1

        for i in range(1, hSize + 1): # Resets the nodes for the next H
            for node in Node.nodesDict[i]:
                node.neighborhoods = []
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

with open('k4k4e_08.g6', 'r') as file:
    k4j4 = file.read().splitlines()
k4j4 = [formatGraph(decodeG6(graph)) for graph in k4j4] # Relevant H's

with open('k3k5e_09.g6', 'r') as file:
    orig = file.read().splitlines()
k3j5 = [formatGraph(decodeG6(graph)) for graph in orig]  # Relevant G's

start = time.time()
gluings = glueG2H(k3j5, 9, k4j4, 8, Node) # Glues G's to H's
print("Total time: ", time.time() - start)
print("Total gluings: ", len(gluings))
"""
for glue in gluings:
    if not isk4j5(glue, 9):
        print("error")

compressedGluings = [compressG6(glue, 9) + "\n" for glue in gluings] # Compresses successful gluings
glueFile = open('glueFileH2G.txt', 'w')
glueFile.writelines(compressedGluings) # Writes gluings to a file
glueFile.close()
"""
