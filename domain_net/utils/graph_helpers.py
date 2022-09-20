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

def get_cell_node_column_names(G, cell_node):
    '''
    Given a graph `G` and a `cell_node` return the unique column names of all its connected attribute nodes
    Note: This function only works for graphs `G` where attribute nodes have a 'column_name' type (i.e. synthetic benchmark graphs) 
    '''
    attribute_nodes = get_attribute_of_instance(G, cell_node)
    column_names = []
    for attr in attribute_nodes:
        column_names.append(G.nodes[attr]['column_name'])
    return list(set(column_names))