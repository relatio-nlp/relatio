# MIT License

# Copyright (c) 2020-2021 ETH Zurich, Andrei V. Plamada
# Copyright (c) 2020-2021 ETH Zurich, Elliott Ash
# Copyright (c) 2020-2021 University of St.Gallen, Philine Widmer
# Copyright (c) 2020-2021 Ecole Polytechnique, Germain Gauthier

# GRAPHS
# ..................................................................................................................
# ..................................................................................................................

import matplotlib.pyplot as plt
import networkx as nx
from pyvis import network as net


def build_graph(  # to be considered as preliminary
    dict_edges,
    dict_args={},
    edge_threshold=0,
    node_threshold=0,
    node_size=None,
    edge_size=None,
    prune_network=True,
):
    # Network specifics
    G = nx.MultiDiGraph()
    if edge_size == None:
        for l in dict_edges:
            G.add_edge(
                l["ARG0"],
                l["ARG1"],
                weight=l["weight"],
                label=l["B-V"],
                title=" ",
                hidden=False,
                color=l["color"],
            )
    else:
        for l in dict_edges:
            G.add_edge(
                l["ARG0"],
                l["ARG1"],
                weight=l["weight"],
                value=l["weight"],
                label=l["B-V"],
                title=" ",
                hidden=False,
                color=l["color"],
            )

    d = nx.degree(G)  # to size nodes according to their degree

    for node in list(G.nodes):
        G.nodes[node]["size"] = node_size
        if node in dict_args:
            G.nodes[node]["color"] = dict_args[node]
        G.nodes[node]["hidden"] = False
        G.nodes[node]["title"] = " "

        # Unlabel infrequent nodes
        if d[node] < node_threshold:
            G.nodes[node]["label"] = " "
            G.nodes[node]["hidden"] = True

    # Hide infrequent edges and edges of infrequent nodes
    for edge in list(G.edges):
        if G.nodes[edge[0]]["hidden"]:
            G.edges[edge]["hidden"] = True
        if G.nodes[edge[1]]["hidden"]:
            G.edges[edge]["hidden"] = True
        if G.edges[edge]["weight"] < edge_threshold:
            G.edges[edge]["hidden"] = True

    if prune_network:
        # Generate connected components and select the largest:
        largest_component = max(nx.weakly_connected_components(G), key=len)

        # Create a subgraph of G consisting only of this component:
        G = G.subgraph(largest_component)

    return G


def draw_graph(
    networkx_graph,
    notebook=True,
    output_filename="graph.html",
    width="1000px",
    height="1000px",
    show_buttons=False,
    only_physics_buttons=False,
):
    """
    This function accepts a networkx graph object,
    converts it to a pyvis network object preserving its node and edge attributes,
    and both returns and saves a dynamic network visualization.

    Valid node attributes include:
        "size", "value", "title", "x", "y", "label", "color".

        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)

    Valid edge attributes include:
        "arrowStrikethrough", "hidden", "physics", "title", "value", "width"

        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_edge)


    Args:
        networkx_graph: The graph to convert and display
        notebook: Display in Jupyter?
        output_filename: Where to save the converted network
        width = width of the network
        height = height of th network
        show_buttons: Show buttons in saved version of network?
        only_physics_buttons: Show only buttons controlling physics of network?
    """

    # make a pyvis network
    pyvis_graph = net.Network(notebook=notebook, directed=True)
    pyvis_graph.width = "1000px"
    pyvis_graph.height = "1000px"

    # for each node and its attributes in the networkx graph
    for node, node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(node, **node_attrs)

    # for each edge and its attributes in the networkx graph
    for source, target, edge_attrs in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(source, target, **edge_attrs)

    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=["physics"])
        else:
            pyvis_graph.show_buttons()

    # Make sure edges aren't written on one another
    pyvis_graph.set_edge_smooth("dynamic")

    # return and also save
    return pyvis_graph.show(output_filename)
