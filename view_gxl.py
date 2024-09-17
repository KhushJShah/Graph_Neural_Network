import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

def read_gxl(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Check and print the root tag to understand the structure
        print("Root tag:", root.tag)
        
        # Find the graph element
        graph = root.find('graph')
        
        if graph is None:
            print("Graph element not found.")
            return
        
        print("Graph ID:", graph.attrib['id'])
        
        nodes = {}
        edges = []

        for node in graph.findall('node'):
            node_id = node.attrib['id']
            attrs = {}
            for attr in node.findall('attr'):
                attr_name = attr.attrib['name']
                attr_value = float(attr.find('float').text)
                attrs[attr_name] = attr_value
            nodes[node_id] = attrs

        print("\nNodes:")
        for node_id, attrs in nodes.items():
            print(f"Node ID: {node_id}, Attributes: {attrs}")

        for edge in graph.findall('edge'):
            from_node = edge.attrib['from']
            to_node = edge.attrib['to']
            edges.append((from_node, to_node))

        print("\nEdges:")
        for from_node, to_node in edges:
            print(f"From: {from_node}, To: {to_node}")

        return nodes, edges

    except ET.ParseError as e:
        print(f"Error parsing the GXL file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def visualize_graph(nodes, edges):
    G = nx.Graph()

    # Add nodes with their attributes
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)

    # Add edges
    for from_node, to_node in edges:
        G.add_edge(from_node, to_node)

    # Extract positions for nodes
    pos = {node_id: (attrs['x'], attrs['y']) for node_id, attrs in nodes.items()}

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold')
    plt.show()

# Example usage
file_path = "C:/Users/nupur/computer/Desktop/ggdlib-main/ggdlib-main/data/Letter/LOW/ZP1_0002.gxl"
nodes, edges = read_gxl(file_path)
visualize_graph(nodes, edges)
