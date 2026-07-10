import networkx as nx
from ._lgpa_core import LGPA_Core

class LGPA:
    """
    Log-Gravity Propagation Algorithm (LGPA) 
    """
    def __init__(self, graph):
        self.G = graph.to_undirected()
        
        self.nodes = list(self.G.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        n = len(self.nodes)
        edges = [(self.node_to_idx[u], self.node_to_idx[v]) for u, v in self.G.edges()]
        
        self.core = LGPA_Core(n, edges)

    def fit_predict(self, max_iter=50):
        if not self.nodes:
            return {}
        
        raw_labels = self.core.fit_predict(max_iter)
        
        return {self.idx_to_node[idx]: label for idx, label in raw_labels.items()}