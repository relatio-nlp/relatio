import networkx as nx
import pandas as pd
from pyvis import network as net


def build_graph(
    narratives,
    top_n,
    prune_network=True,
):
    """
    A function that builds a networkx graph from a list of narratives.

    Args:
        narratives: a list of narratives
        top_n: number of most frequent narratives to include in the graph
        prune_network: whether to prune the network to its largest component

    Returns:
        a networkx graph of the most frequent narratives
    """

    # Network specifics
    G = nx.MultiDiGraph()

    df = pd.DataFrame(narratives)
    df = df.replace(False, "")
    df = df.replace(True, "not ")
    df = df.replace(pd.NA, "")
    df = df.value_counts().reset_index(name="counts")
    df = df[df["ARG0"] != ""]
    df = df[df["ARG1"] != ""]
    df = df[df["B-V"] != ""]
    df = df.reset_index()

    temp = df.to_dict(orient="records")[0:top_n]

    for l in temp:
        G.add_edge(
            l["ARG0"],
            l["ARG1"],
            value=l["counts"],
            label=l.get("B-ARGM-NEG", "") + l["B-V"],
            hidden=False,
        )

    d = nx.degree(G)
    for i, node in enumerate(list(G.nodes)):
        G.nodes[node]["value"] = d[node]

    if prune_network:
        largest_component = max(nx.weakly_connected_components(G), key=len)
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
    pyvis_graph.width = width
    pyvis_graph.height = height

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
