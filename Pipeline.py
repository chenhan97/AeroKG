import networkx as nx
import matplotlib.pyplot as plt

def visulization(triplets):
    # triplets format: [[ent1, ent2, rel],...]
    input_list = [[i[0],i[1],{"rel":i[2]}] for i in triplets]
    G = nx.MultiDiGraph()
    G.add_edges_from(input_list)
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
    nx.draw_networkx_edge_labels(G, pos=pos)
    plt.show()
