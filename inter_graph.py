import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

def read_gxl(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        graph = root.find('graph')
        if graph is None:
            print("Graph element not found.")
            return None, None

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

        for edge in graph.findall('edge'):
            from_node = edge.attrib['from']
            to_node = edge.attrib['to']
            edges.append((from_node, to_node))

        return nodes, edges

    except ET.ParseError as e:
        print(f"Error parsing the GXL file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def find_intersections_and_add_nodes(nodes, edges):
    G = nx.Graph()

    # Add nodes with attributes
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)

    # Add edges
    for from_node, to_node in edges:
        G.add_edge(from_node, to_node)

    new_nodes = {}
    new_edges = []
    node_count = len(nodes)

    # Check for intersections between all pairs of edges
    for i, (from_node1, to_node1) in enumerate(edges):
        line1 = LineString([(nodes[from_node1]['x'], nodes[from_node1]['y']),
                            (nodes[to_node1]['x'], nodes[to_node1]['y'])])

        for j, (from_node2, to_node2) in enumerate(edges):
            if i >= j:  # Avoid checking the same pair twice or comparing the same edge
                continue

            line2 = LineString([(nodes[from_node2]['x'], nodes[from_node2]['y']),
                                (nodes[to_node2]['x'], nodes[to_node2]['y'])])

            # Check for intersection
            intersection = line1.intersection(line2)
            if isinstance(intersection, Point):
                # Check if the intersection point already exists as a node
                existing_node = None
                for node_id, attrs in nodes.items():
                    if attrs['x'] == intersection.x and attrs['y'] == intersection.y:
                        existing_node = node_id
                        break

                if existing_node:
                    new_node_id = existing_node  # Use the existing node
                else:
                    new_node_id = f"_new_{node_count}"  # Create a new node
                    node_count += 1
                    new_nodes[new_node_id] = {'x': intersection.x, 'y': intersection.y}
                    G.add_node(new_node_id, x=intersection.x, y=intersection.y)

                if from_node1 != new_node_id and to_node1 != new_node_id:
                    new_edges.append((from_node1, new_node_id))
                    new_edges.append((new_node_id, to_node1))

                if from_node2 != new_node_id and to_node2 != new_node_id:
                    new_edges.append((from_node2, new_node_id))
                    new_edges.append((new_node_id, to_node2))


    # Add new nodes and edges to the graph without modifying the original ones
    for node_id, attrs in new_nodes.items():
        nodes[node_id] = attrs
    edges.extend(new_edges)

    return nodes, edges

def visualize_graph(nodes, edges):
    G = nx.Graph()

    # Add nodes with attributes
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

file_path = "C:/Users/nupur/computer/Desktop/ggdlib-main/ggdlib-main/data/Letter/LOW/AP1_0002.gxl"
nodes, edges = read_gxl(file_path)
if nodes and edges:
    nodes, edges = find_intersections_and_add_nodes(nodes, edges)
    visualize_graph(nodes, edges)