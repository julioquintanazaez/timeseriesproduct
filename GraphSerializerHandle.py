from fastapi import HTTPException, Query, File, UploadFile
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import Generator, Dict, Any, List, Tuple
import numpy as np
import networkx as nx
from datetime import datetime
import logging

from datahandle import convert_numpy_nativo

logger = logging.getLogger(__name__)


class GraphStreamSerializer:
    """Serializes graph data as a stream without pagination"""
    
    def __init__(self, G: nx.MultiGraph, analyses_results: List, advanced_analysis: Dict, 
                 stationarity_clusters: Dict, parameters: Dict, constant_products: List):
        self.G = G
        self.analyses_results = analyses_results
        self.advanced_analysis = advanced_analysis
        self.stationarity_clusters = stationarity_clusters
        self.parameters = parameters
        self.constant_products = constant_products
    
    def stream_full_response(self) -> Generator[str, None, None]:
        """Streams complete response including statistics and graph data"""
        
        # 1. Start JSON object
        yield '{"status":"success",\n'
        
        # 2. Parameters section
        yield f'"parameters": {json.dumps(self.parameters, default=str)},\n'
        
        # 3. Statistics section (from advanced_analysis)
        yield f'"statistics": {self._serialize_statistics()},\n'
        
        # 4. Stationarity clusters
        yield f'"stationarity_clusters": {self._serialize_clusters()},\n'
        
        # 5. Constant products excluded
        yield f'"constant_products_excluded": {json.dumps(self.constant_products[:50])},\n'
        
        # 6. Graph data section
        yield '"graph_data": {\n'
        
        # 7. Graph summary
        yield f'  "summary": {self._serialize_summary()},\n'
        
        # 8. Nodes
        yield '  "nodes": [\n'
        yield from self._stream_nodes()
        yield '  ],\n'
        
        # 9. Edges
        yield '  "edges": [\n'
        yield from self._stream_edges()
        yield '  ]\n'
        
        # 10. Close graph_data and main object
        yield '}\n'
        yield '}'
    
    def _serialize_statistics(self) -> str:
        """Serializes statistics as JSON string"""
        # Convert numpy types to Python native
        stats = convert_numpy_nativo(self.advanced_analysis)
        return json.dumps(stats, default=str)
    
    def _serialize_clusters(self) -> str:
        """Serializes stationarity clusters"""
        clusters = {
            "stationary": [str(p) for p in self.stationarity_clusters.get('stationary', [])],
            "non_stationary": [str(p) for p in self.stationarity_clusters.get('non_stationary', [])],
            "stationary_count": len(self.stationarity_clusters.get('stationary', [])),
            "non_stationary_count": len(self.stationarity_clusters.get('non_stationary', []))
        }
        return json.dumps(clusters, default=str)
    
    def _serialize_summary(self) -> str:
        """Serializes graph summary"""
        summary = {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "product_nodes": sum(1 for n, d in self.G.nodes(data=True) if d.get('node_type') == 'product'),
            "store_nodes": sum(1 for n, d in self.G.nodes(data=True) if d.get('node_type') == 'store'),
            "relation_counts": self._get_relation_counts(),
            "connected_components": nx.number_connected_components(self.G),
            "density": nx.density(self.G) if self.G.number_of_nodes() > 0 else 0
        }
        return json.dumps(summary, default=str)
    
    def _get_relation_counts(self) -> Dict:
        """Counts relation types"""
        counts = {}
        for _, _, data in self.G.edges(data=True):
            rel_type = data.get('relation_type', 'unknown')
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts
    
    def _stream_nodes(self) -> Generator[str, None, None]:
        """Streams nodes one by one"""
        nodes_list = list(self.G.nodes(data=True))
        
        for i, (node, attrs) in enumerate(nodes_list):
            node_data = {
                'id': str(node),
                'type': attrs.get('node_type', 'unknown'),
                'properties': self._extract_node_properties(attrs)
            }
            
            node_json = json.dumps(node_data, default=str)
            yield f'    {node_json}'
            
            if i < len(nodes_list) - 1:
                yield ',\n'
            else:
                yield '\n'
    
    def _stream_edges(self) -> Generator[str, None, None]:
        """Streams edges one by one"""
        edges_list = list(self.G.edges(data=True))
        
        for i, (u, v, data) in enumerate(edges_list):
            edge_data = {
                'source': str(u),
                'target': str(v),
                'relation_type': data.get('relation_type', 'unknown'),
                'weight': float(data.get('weight', 1.0)),
                'properties': self._extract_edge_properties(data)
            }
            
            edge_json = json.dumps(edge_data, default=str)
            yield f'    {edge_json}'
            
            if i < len(edges_list) - 1:
                yield ',\n'
            else:
                yield '\n'
    
    @staticmethod
    def _extract_node_properties(attrs: Dict) -> Dict:
        """Extracts node properties"""
        if attrs.get('node_type') == 'product':
            return {
                'product_name': attrs.get('product_name', str(attrs.get('id', ''))),
                'is_stationary': bool(attrs.get('is_stationary', False)),
                'stationarity_pvalue': float(attrs.get('stationarity_pvalue', 0)) if attrs.get('stationarity_pvalue') else None,
                'trend_direction': attrs.get('trend_direction', 'unknown'),
                'trend_slope': float(attrs.get('trend_slope', 0)),
                'trend_magnitude': float(attrs.get('trend_magnitude', 0)),
                'mean_sales': float(attrs.get('mean_sales', 0)),
                'cv': float(attrs.get('cv', 0)),
                'main_period': int(attrs.get('main_period', 0))
            }
        return {'store_id': str(attrs.get('store_id', ''))}
    
    @staticmethod
    def _extract_edge_properties(data: Dict) -> Dict:
        """Extracts edge properties"""
        properties = {}
        if data.get('relation_type') == 'PRODUCT_SIMILARITY':
            properties['same_stationarity'] = bool(data.get('same_stationarity', False))
            properties['slope_diff'] = float(data.get('slope_diff', 0))
            properties['same_trend_direction'] = bool(data.get('same_trend_direction', False))
        elif data.get('relation_type') == 'SAME_STORE':
            properties['sales_count'] = int(data.get('sales_count', 0))
        elif data.get('relation_type') == 'PURCHASED_TOGETHER':
            properties['times_together'] = int(data.get('times_together', 0))
            properties['lift'] = float(data.get('lift', 0))
            properties['confidence'] = float(data.get('confidence', 0))
        return properties

