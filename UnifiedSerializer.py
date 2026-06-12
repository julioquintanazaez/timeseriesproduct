# unified_serializer.py
import json
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Generator, Any, Tuple
import networkx as nx
import pandas as pd

# unified_serializer.py
import json
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Generator, Any
import networkx as nx
import pandas as pd

import logging

logger = logging.getLogger(__name__)

class UnifiedGraphSerializer:
    """
    Serializes complete graph analysis with sparse matrix representation.
    Handles AnalysisResult objects correctly.
    """
    
    def __init__(self, G: nx.MultiGraph, analyses_results: List, advanced_analysis: Dict,
                 stationarity_clusters: Dict, parameters: Dict, constant_products: List):
        self.G = G
        self.analyses_results = analyses_results
        self.advanced_analysis = advanced_analysis
        self.stationarity_clusters = stationarity_clusters
        self.parameters = parameters
        self.constant_products = constant_products
        
        # Mapear nodos a índices para matriz dispersa
        self.nodes_list = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
        
        # Construir matrices dispersas
        self._build_sparse_matrices()
        
        # Crear índice de análisis por producto
        self._build_analyses_index()
    
    def _build_sparse_matrices(self):
        """Builds separate sparse matrices for each relation type"""
        from collections import defaultdict
        
        edges_by_type = defaultdict(lambda: {'rows': [], 'cols': [], 'weights': [], 'properties': []})
        
        for u, v, data in self.G.edges(data=True):
            rel_type = data.get('relation_type', 'unknown')
            u_idx = self.node_to_idx.get(u)
            v_idx = self.node_to_idx.get(v)
            
            if u_idx is None or v_idx is None:
                continue  # Skip if node not in mapping
                
            weight = float(data.get('weight', 1.0))
            
            edges_by_type[rel_type]['rows'].append(u_idx)
            edges_by_type[rel_type]['cols'].append(v_idx)
            edges_by_type[rel_type]['weights'].append(weight)
            
            props = self._extract_edge_properties(data)
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
    
    def _extract_edge_properties(self, data: Dict) -> Dict:
        """Extracts properties based on relation type"""
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
    
    def _build_analyses_index(self):
        """Creates index to quickly find product analysis by product ID"""
        self.product_analyses_index = {}
        
        for analysis in self.analyses_results:
            # Handle both dictionary and object
            if hasattr(analysis, 'product_id'):
                product_id = str(analysis.product_id)
                self.product_analyses_index[product_id] = analysis
            elif isinstance(analysis, dict):
                product_id = str(analysis.get('product_id'))
                if product_id:
                    self.product_analyses_index[product_id] = analysis
            else:
                # Try to get product_id attribute or key
                try:
                    product_id = str(getattr(analysis, 'product_id', None))
                    if product_id and product_id != 'None':
                        self.product_analyses_index[product_id] = analysis
                except:
                    pass
    
    def _extract_analysis_dict(self, analysis) -> Dict:
        """Extract dictionary from AnalysisResult object or dict"""
        if hasattr(analysis, '__dict__'):
            # Convert object to dict
            result = {}
            for key in ['product_id', 'store_id', 'is_stationary', 'stationarity_pvalue',
                       'trend_direction', 'trend_slope', 'trend_magnitude', 
                       'mean_sales', 'cv', 'main_period']:
                if hasattr(analysis, key):
                    value = getattr(analysis, key)
                    if value is not None:
                        if isinstance(value, (np.integer, np.floating)):
                            value = value.item()
                        result[key] = value
            return result
        elif isinstance(analysis, dict):
            return analysis
        else:
            return {}
    
    def export_unified(self) -> Dict[str, Any]:
        """Exports complete unified data structure"""
        
        unified_data = {
            "format": "unified_sparse_graph",
            "version": "2.0",
            "timestamp": str(pd.Timestamp.now()),
            
            # Parámetros
            "parameters": self._serialize_parameters(),
            
            # Estadísticas
            "statistics": self._serialize_statistics(),
            
            # Clusters
            "stationarity_clusters": self._serialize_clusters(),
            
            # Productos excluidos
            "constant_products_excluded": self.constant_products[:100],
            
            # Análisis de productos
            "product_analyses": self._serialize_product_analyses(),
            
            # Matriz dispersa
            "sparse_graph": self._serialize_sparse_graph(),
            
            # Resumen rápido
            "quick_summary": self._quick_summary()
        }
        
        return unified_data
    
    def _serialize_parameters(self) -> Dict:
        """Serializes parameters safely"""
        params = {}
        safe_keys = ['frequency', 'stationarity_threshold', 'trend_similarity_threshold',
                    'min_purchases_together', 'include_store_nodes', 'include_product_nodes',
                    'total_products_analyzed', 'total_products_filtered']
        
        for key in safe_keys:
            value = self.parameters.get(key)
            if value is not None:
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                params[key] = value
        return params
    
    def _serialize_statistics(self) -> Dict:
        """Serializes advanced analysis statistics"""
        stats = {}
        for key, value in self.advanced_analysis.items():
            if isinstance(value, np.generic):
                stats[key] = value.item()
            elif isinstance(value, np.ndarray):
                stats[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                stats[key] = value
            else:
                stats[key] = str(value)
        return stats
    
    def _serialize_clusters(self) -> Dict:
        """Serializes stationarity clusters"""
        stationary = self.stationarity_clusters.get('stationary', [])
        non_stationary = self.stationarity_clusters.get('non_stationary', [])
        
        return {
            "stationary": [str(p) for p in stationary],
            "non_stationary": [str(p) for p in non_stationary],
            "stationary_count": len(stationary),
            "non_stationary_count": len(non_stationary),
            "stationary_percentage": (len(stationary) / max(1, len(self.analyses_results))) * 100
        }
    
    def _serialize_product_analyses(self) -> List[Dict]:
        """Serializes all product analyses"""
        analyses = []
        
        for analysis in self.analyses_results[:5000]:  # Limit to prevent huge payload
            analysis_dict = self._extract_analysis_dict(analysis)
            
            # Only include essential fields
            simplified = {
                "product_id": str(analysis_dict.get('product_id', '')),
                "store_id": str(analysis_dict.get('store_id', '')),
                "is_stationary": bool(analysis_dict.get('is_stationary', False)),
                "stationarity_pvalue": float(analysis_dict.get('stationarity_pvalue', 0)) if analysis_dict.get('stationarity_pvalue') is not None else None,
                "trend_direction": str(analysis_dict.get('trend_direction', 'unknown')),
                "trend_slope": float(analysis_dict.get('trend_slope', 0)),
                "trend_magnitude": float(analysis_dict.get('trend_magnitude', 0)),
                "mean_sales": float(analysis_dict.get('mean_sales', 0)),
                "cv": float(analysis_dict.get('cv', 0)),
                "main_period": int(analysis_dict.get('main_period', 0)) if analysis_dict.get('main_period') else None
            }
            analyses.append(simplified)
        
        return analyses
    
    def _serialize_sparse_graph(self) -> Dict:
        """Serializes sparse matrix representation"""
        
        # Metadata de nodos
        node_metadata = []
        for idx, node in enumerate(self.nodes_list):
            attrs = self.G.nodes[node]
            
            if attrs.get('node_type') == 'product':
                product_id = str(node)
                product_analysis = self.product_analyses_index.get(product_id)
                
                metadata = {
                    'id': product_id,
                    'node_type': 'product',
                    'index': idx,
                    'product_name': str(attrs.get('product_name', product_id)),
                    'store_id': str(attrs.get('store_id', ''))
                }
                
                # Añadir análisis si está disponible
                if product_analysis:
                    analysis_dict = self._extract_analysis_dict(product_analysis)
                    metadata.update({
                        'is_stationary': bool(analysis_dict.get('is_stationary', False)),
                        'stationarity_pvalue': float(analysis_dict.get('stationarity_pvalue', 0)) if analysis_dict.get('stationarity_pvalue') else None,
                        'trend_direction': str(analysis_dict.get('trend_direction', 'unknown')),
                        'trend_slope': float(analysis_dict.get('trend_slope', 0)),
                        'trend_magnitude': float(analysis_dict.get('trend_magnitude', 0)),
                        'mean_sales': float(analysis_dict.get('mean_sales', 0)),
                        'cv': float(analysis_dict.get('cv', 0)),
                        'main_period': int(analysis_dict.get('main_period', 0)) if analysis_dict.get('main_period') else None
                    })
                else:
                    # Usar atributos del nodo como fallback
                    metadata.update({
                        'is_stationary': bool(attrs.get('is_stationary', False)),
                        'stationarity_pvalue': float(attrs.get('stationarity_pvalue', 0)) if attrs.get('stationarity_pvalue') else None,
                        'trend_direction': str(attrs.get('trend_direction', 'unknown')),
                        'trend_slope': float(attrs.get('trend_slope', 0)),
                        'trend_magnitude': float(attrs.get('trend_magnitude', 0)),
                        'mean_sales': float(attrs.get('mean_sales', 0)),
                        'cv': float(attrs.get('cv', 0)),
                        'main_period': int(attrs.get('main_period', 0)) if attrs.get('main_period') else None
                    })
            else:
                metadata = {
                    'id': str(node),
                    'node_type': 'store',
                    'index': idx,
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
                'shape': list(matrix.shape),
                'nnz': int(matrix.nnz)
            }
        
        # Contar relaciones
        relation_counts = {}
        for _, _, data in self.G.edges(data=True):
            rel_type = data.get('relation_type', 'unknown')
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        
        return {
            'total_nodes': len(self.nodes_list),
            'total_edges': self.G.number_of_edges(),
            'nodes': [str(node) for node in self.nodes_list],
            'node_metadata': node_metadata,
            'sparse_matrices': matrices_serializable,
            'matrix_properties': self.matrix_properties,
            'graph_summary': {
                'product_nodes': sum(1 for d in node_metadata if d['node_type'] == 'product'),
                'store_nodes': sum(1 for d in node_metadata if d['node_type'] == 'store'),
                'relation_counts': relation_counts,
                'connected_components': nx.number_connected_components(self.G),
                'density': float(nx.density(self.G)) if len(self.nodes_list) > 0 else 0,
                'is_directed': self.G.is_directed(),
                'is_multigraph': self.G.is_multigraph()
            }
        }
    
    def _quick_summary(self) -> Dict:
        """Provides quick summary statistics"""
        stationary_count = len(self.stationarity_clusters.get('stationary', []))
        non_stationary_count = len(self.stationarity_clusters.get('non_stationary', []))
        
        # Top products by sales
        top_products = []
        for analysis in self.analyses_results[:10]:
            analysis_dict = self._extract_analysis_dict(analysis)
            top_products.append({
                'product_id': str(analysis_dict.get('product_id', '')),
                'store_id': str(analysis_dict.get('store_id', '')),
                'mean_sales': float(analysis_dict.get('mean_sales', 0)),
                'trend_direction': str(analysis_dict.get('trend_direction', 'unknown'))
            })
        
        return {
            'total_products': len(self.analyses_results),
            'stationary_products': stationary_count,
            'non_stationary_products': non_stationary_count,
            'constant_products_excluded': len(self.constant_products),
            'total_stores': self.G.number_of_nodes() - len(self.analyses_results),
            'graph_density': float(nx.density(self.G)) if self.G.number_of_nodes() > 0 else 0,
            'top_10_products_by_sales': top_products
        }
    
    def stream_unified_response(self) -> Generator[str, None, None]:
        """Streams unified response as JSON"""
        
        yield '{\n'
        yield '  "status": "success",\n'
        
        unified_data = self.export_unified()
        
        first_chunk = True
        for key, value in unified_data.items():
            if not first_chunk:
                yield ',\n'
            first_chunk = False
            
            # Handle numpy types
            try:
                json_chunk = json.dumps(value, default=self._json_default)
                yield f'  "{key}": {json_chunk}'
            except Exception as e:
                logger.error(f"Error serializing {key}: {str(e)}")
                yield f'  "{key}": {{"error": "Serialization failed"}}'
        
        yield '\n}'
    
    @staticmethod
    def _json_default(obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def export_compressed(self, compression_level: int = 6) -> bytes:
        """Export as compressed bytes"""
        import gzip
        
        unified_data = self.export_unified()
        json_str = json.dumps(unified_data, default=self._json_default)
        compressed = gzip.compress(json_str.encode('utf-8'), compresslevel=compression_level)
        
        return compressed