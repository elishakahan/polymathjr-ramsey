from sage.graphs.graph_input import from_graph6

with open('glueFile.txt', 'r') as file:
    k4j5 = file.read().splitlines()

sageGraphs = [Graph() for i in range(len(k4j5))]
for i, graph in enumerate(k4j5):
    from_graph6(sageGraphs[i], graph)

newSageGraphs = []
for graph in sageGraphs:
    for graph2 in newSageGraphs:
        if graph.is_isomorphic(graph2):
            break
    else:
        newSageGraphs.append(graph)

for graph in newSageGraphs:
    graph.show()
    print(graph.graph6_string())
