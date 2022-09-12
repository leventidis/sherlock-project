import networkx as nx

def get_attribute_of_instance(G, instance_node):
    '''
    Given a graph `G` and an `instance_node` from the graph return its corresponding set of attribute nodes
    '''
    attribute_nodes = []
    for neighbor in G[instance_node]:
        if G.nodes[neighbor]['type'] == 'attr':
            attribute_nodes.append(neighbor)
    return attribute_nodes