# serializer.py
import json
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Generator, Any
import networkx as nx

from datahandle import convert_numpy_nativo

class SparseMatrixSerializer:
    """Serializes graph as sparse adjacency matrix (much more compact for large graphs)"""
    
    def __init__(self, G: nx.MultiGraph, analyses_results: List, advanced_analysis: Dict,
                 stationarity_clusters: Dict, parameters: Dict, constant_products: List):
        self.G = G
        self.analyses_results = analyses_results
        self.advanced_analysis = advanced_analysis
        self.stationarity_clusters = stationarity_clusters
        self.parameters = parameters
        self.constant_products = constant_products
        
        # Mapear nodos a índices
        self.nodes_list = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
        
        # Construir matrices por tipo de relación
        self._build_sparse_matrices()
    
    def _build_sparse_matrices(self):
        """Builds separate sparse matrices for each relation type"""
        from collections import defaultdict
        
        # Diccionario para almacenar edges por tipo de relación
        edges_by_type = defaultdict(lambda: {'rows': [], 'cols': [], 'weights': [], 'properties': []})
        
        for u, v, data in self.G.edges(data=True):
            rel_type = data.get('relation_type', 'unknown')
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            weight = float(data.get('weight', 1.0))
            
            edges_by_type[rel_type]['rows'].append(u_idx)
            edges_by_type[rel_type]['cols'].append(v_idx)
            edges_by_type[rel_type]['weights'].append(weight)
            
            # Guardar propiedades adicionales según tipo de relación
            props = self._extract_edge_properties_for_matrix(data)
            edges_by_type[rel_type]['properties'].append(props)
        
        # Convertir a matrices CSR
        self.sparse_matrices = {}
        self.matrix_properties = {}
        
        for rel_type, data in edges_by_type.items():
            if data['rows']:
                matrix = csr_matrix(
                    (data['weights'], (data['rows'], data['cols'])),
                    shape=(len(self.nodes_list), len(self.nodes_list))
                )
                self.sparse_matrices[rel_type] = matrix
                self.matrix_properties[rel_type] = data['properties']
            else:
                self.sparse_matrices[rel_type] = csr_matrix((len(self.nodes_list), len(self.nodes_list)))
                self.matrix_properties[rel_type] = []
    
    def _extract_edge_properties_for_matrix(self, data: Dict) -> Dict:
        """Extracts relevant properties for matrix format"""
        rel_type = data.get('relation_type', 'unknown')
        
        if rel_type == 'PRODUCT_SIMILARITY':
            return {
                'same_stationarity': bool(data.get('same_stationarity', False)),
                'slope_diff': float(data.get('slope_diff', 0)),
                'same_trend_direction': bool(data.get('same_trend_direction', False))
            }
        elif rel_type == 'SAME_STORE':
            return {
                'sales_count': int(data.get('sales_count', 0))
            }
        elif rel_type == 'PURCHASED_TOGETHER':
            return {
                'times_together': int(data.get('times_together', 0)),
                'lift': float(data.get('lift', 0)),
                'confidence': float(data.get('confidence', 0))
            }
        return {}
    
    def export_to_sparse_matrix(self) -> Dict[str, Any]:
        """Exports complete graph as sparse matrix representation"""
        
        # Preparar metadata de nodos
        node_metadata = []
        for node in self.nodes_list:
            attrs = self.G.nodes[node]
            if attrs.get('node_type') == 'product':
                metadata = {
                    'id': str(node),
                    'node_type': 'product',
                    'product_name': attrs.get('product_name', str(node)),
                    'is_stationary': bool(attrs.get('is_stationary', False)),
                    'stationarity_pvalue': float(attrs.get('stationarity_pvalue', 0)) if attrs.get('stationarity_pvalue') else None,
                    'trend_direction': attrs.get('trend_direction', 'unknown'),
                    'trend_slope': float(attrs.get('trend_slope', 0)),
                    'trend_magnitude': float(attrs.get('trend_magnitude', 0)),
                    'mean_sales': float(attrs.get('mean_sales', 0)),
                    'cv': float(attrs.get('cv', 0)),
                    'main_period': int(attrs.get('main_period', 0))
                }
            else:
                metadata = {
                    'id': str(node),
                    'node_type': 'store',
                    'store_id': str(attrs.get('store_id', ''))
                }
            node_metadata.append(metadata)
        
        # Convertir matrices a formato serializable
        matrices_serializable = {}
        for rel_type, matrix in self.sparse_matrices.items():
            matrices_serializable[rel_type] = {
                'indices': matrix.indices.tolist(),
                'indptr': matrix.indptr.tolist(),
                'data': matrix.data.tolist(),
                'shape': matrix.shape
            }
        
        return {
            'format': 'sparse_matrix',
            'version': '1.0',
            'total_nodes': len(self.nodes_list),
            'total_edges': self.G.number_of_edges(),
            'nodes': self.nodes_list,  # Lista de IDs de nodos
            'node_metadata': node_metadata,
            'sparse_matrices': matrices_serializable,
            'matrix_properties': self.matrix_properties,
            'graph_summary': {
                'product_nodes': sum(1 for d in node_metadata if d['node_type'] == 'product'),
                'store_nodes': sum(1 for d in node_metadata if d['node_type'] == 'store'),
                'relation_counts': dict(self._get_relation_counts()),
                'connected_components': nx.number_connected_components(self.G),
                'density': nx.density(self.G) if len(self.nodes_list) > 0 else 0
            }
        }
    
    def _get_relation_counts(self):
        """Counts relation types"""
        counts = {}
        for _, _, data in self.G.edges(data=True):
            rel_type = data.get('relation_type', 'unknown')
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts
    
    def stream_sparse_matrix(self) -> Generator[str, None, None]:
        """Streams sparse matrix representation as JSON"""
        
        # 1. Iniciar objeto JSON
        yield '{"status":"success",\n'
        
        # 2. Parámetros
        yield f'"parameters": {json.dumps(self.parameters, default=str)},\n'
        
        # 3. Estadísticas
        stats = convert_numpy_nativo(self.advanced_analysis)
        yield f'"statistics": {json.dumps(stats, default=str)},\n'
        
        # 4. Clusters
        clusters = {
            "stationary": [str(p) for p in self.stationarity_clusters.get('stationary', [])],
            "non_stationary": [str(p) for p in self.stationarity_clusters.get('non_stationary', [])],
            "stationary_count": len(self.stationarity_clusters.get('stationary', [])),
            "non_stationary_count": len(self.stationarity_clusters.get('non_stationary', []))
        }
        yield f'"stationarity_clusters": {json.dumps(clusters, default=str)},\n'
        
        # 5. Productos constantes
        yield f'"constant_products_excluded": {json.dumps(self.constant_products[:50])},\n'
        
        # 6. Matriz dispersa
        sparse_data = self.export_to_sparse_matrix()
        yield f'"sparse_graph": {json.dumps(sparse_data, default=str)}\n'
        
        # 7. Cerrar objeto
        yield '}'