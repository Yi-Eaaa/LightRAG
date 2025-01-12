import networkx as nx
from pyvis.network import Network

# Load the GraphML file
G = nx.read_graphml('/home/hongyi/LightRAG/dickens/graph_chunk_entity_relation.graphml')

# Create a Pyvis network
net = Network(notebook=True, cdn_resources='in_line')

# Convert NetworkX graph to Pyvis network
net.from_nx(G)

# Save and display the network
with open("/home/hongyi/LightRAG/dickens/index.html", 'a') as f:
    f.write('''<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=/knowledge_graph.html"></head><body></body></html>''')
net.show('/home/hongyi/LightRAG/dickens/knowledge_graph.html')