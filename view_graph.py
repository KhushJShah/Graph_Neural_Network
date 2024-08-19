import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

def parse_gxl(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    graph = nx.Graph()
    
    for node in root.findall('.//node'):
        node_id = node.get('id')
        graph.add_node(node_id)
        
        for attr in node.findall('.//attr'):
            name = attr.get('name')
            value_elem = attr.find('string') or attr.find('int') or attr.find('float')  # Check for different types
            if value_elem is not None:
                value = value_elem.text
                graph.nodes[node_id][name] = value
    
    for edge in root.findall('.//edge'):
        source = edge.get('from')
        target = edge.get('to')
        graph.add_edge(source, target)
        
        for attr in edge.findall('.//attr'):
            name = attr.get('name')
            value_elem = attr.find('string') or attr.find('int') or attr.find('float')  # Check for different types
            if value_elem is not None:
                value = value_elem.text
                graph.edges[source, target][name] = value
    
    return graph

def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw_shell(graph, with_labels=True)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()

file_path = "C:/Users/nupur/computer/Desktop/ggdlib-main/ggdlib-main/data/Letter/LOW/AP1_0001.gxl"
graph = parse_gxl(file_path)
visualize_graph(graph)
