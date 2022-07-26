from sage.graphs.graph_input import from_graph6
from sage.all_cmdline import *

def isEdge(u, v):
    x = u[0] - v[0]; y = u[1] - v[1]; z = u[2] - v[2]
    return ((x != 0) and (x*z) == (y*y)) 
    
def kopparty(q):
    '''
    Returns Kopparty Graph defined over F_q
    '''
    try:
        k = GF(q, repr = 'int') # This gives us the elements of the finite fields as integers
    except:
        raise Exception("The order of a finite field must be a prime power")
    V = list(VectorSpace(k, 3)) # This gives us F^3_q
    for vec in V: 
        vec.set_immutable()
    return Graph([V, lambda u, v: isEdge(u, v)])

with open('r35_13.g6', 'r') as file: # Opens the file, and closes it once we exit this block. 
    graphs = file.readlines() 
r35_13 = Graph() # Create an empty graph
from_graph6(r35_13, graphs[0])
r35_13.show()



def isZero(v): 
    '''
    Function to recognize if v is the zero vector over F_q.
    Surely Sage has a function for this, but I couldn't find it
    '''
    for e in v:
        if e!= 0:
            return False
    return True

def oneDimSubspace(q, dim):
    try:
        k = GF(q, repr = 'int') # This gives us the elements of the finite fields as integers
    except:
        raise Exception("The order of a finite field must be a prime power")
    V = list(VectorSpace(k, dim))
    for v in V: # Ugly hack to remove the zero vector from V
        if isZero(v):
            V.remove(v)
            break
    scal_list = [x for _, x in enumerate(k) if (x != 0 and x != 1)] #Non zero, non-one elements of F_q
    for vec in V:
        '''
        For each vector, we remove all of it's scalar multiples from V.
        The vectors which remain are considered to be representatives of the 1 dimensional subspaces they generate
        '''
        for s in scal_list:
            if vec*s in V:
                V.remove(vec*s)
    for vec in V: vec.set_immutable()
    return V



def beta(u, v):
    if len(u) != len(v) or len(u) % 2 != 0: 
        raise Exception('Input vectors need to have equal and even dimension.')
    
    B = 0
    for i in range(0, len(u), 2): 
        B += u[i] * v[i + 1] - u[i + 1] * v[i]
    
    return B

def kopparty_alt(q):
    V = oneDimSubspace(q, 4)
    return Graph([V, lambda u, v: (u != v and beta(u, v) == 0)])

def dot(u, v):
    if len(u) != len(v): 
        raise Exception('Input vectors need to have equal dimensions')
    B = 0
    for i in range(len(u)):
        B += u[i]*v[i]
    return B    

def filt(v):
    retval = []
    for x in v:
        if dot(x, x) != 0:
            retval.append(x)
    return retval

def alon_kriv(q,s):
    V = filt(oneDimSubspace(q,s-1))
    return Graph([V, lambda u, v: dot(u, v) == 0])


kp9 = kopparty(9)

subg3 = kp9.subgraph_search(r35_13, induced=True)
if subg3 is not None:
    print(subg3.vertices())



kpa8 = kopparty_alt(8)


s1 = kpa8.subgraph_search(r35_13) #This takes quite some time (4-5 minutes)
if s1 is not None:
    print(s1.vertices())


print("Graph statistics")
print("Number of vertices:", kp9.order())
print("Number of Edges:", kp9.size())
print("Degree Sequence:", kp9.degree_sequence())
print("Spectrum", kp9.spectrum())
print("Number of vertices:", kpa8.order())
print("Number of Edges:", kpa8.size())
print("Degree Sequence:", kpa8.degree_sequence())
print("Spectrum", kpa8.spectrum())