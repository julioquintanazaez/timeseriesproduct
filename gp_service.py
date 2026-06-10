import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import combinations
from collections import defaultdict

import networkx as nx
import logging

from models import ProductSeries

from ts_service import (
    analyze_all_products_optimized,
    create_feature_matrix_optimized,
    normalize_features_optimized,
    perform_kmeans_optimized,
    perform_dbscan_optimized,
    perform_hierarchical_optimized,
    extract_cluster_profiles_optimized,
    analyze_product_optimized,
    aggregate_sales_by_time_optimized,
    extract_product_series_optimized
)

from ts_service import AnalysisResult

from datehandle import normalizar_fechas, normalize_dates_optimized, normalize_dates_vectorized

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def convertir_columna_datetime(df, columna):
    """
    Convierte una columna datetime64 de pandas a datetime nativo de Python.
    Modifica el DataFrame in-place.
    """
    if columna in df.columns and pd.api.types.is_datetime64_any_dtype(df[columna]):
        # Convertir la serie completa de una vez
        df[columna] = df[columna].dt.to_pydatetime()
    return df


def aggregate_sales_by_product_store_optimized(
    sales_df: pd.DataFrame, 
    frequency: str = 'D'
) -> pd.DataFrame:
    """
    Aggregates sales data by product-store combination, creating time series
    for each unique (product_id, store_id) pair.
    
    Returns DataFrame with columns: product_store_key, id_product, product_name, 
    id_tienda, time_sale, sales_count, avg_amount, amount_std, min_amount, max_amount
    """
    
    sales_df = sales_df.copy()
    
    print("Parsing time...")
    # Convert to datetime
    sales_df['time_sale'] = pd.to_datetime(sales_df['time_sale'], errors='coerce')
    sales_df = sales_df.dropna(subset=['time_sale'])
    
    if sales_df.empty:
        raise ValueError("No valid dates found in time_sale column")
    
    # Create floor date for aggregation
    sales_df['date'] = sales_df['time_sale'].dt.floor(frequency)
    
    # Create unique product-store identifier
    sales_df['product_store_key'] = sales_df['id_product'].astype(str) + '_' + sales_df['id_tienda'].astype(str)
    
    # Get product names
    product_names = sales_df.groupby('id_product')['name'].first()
    
    print("Aggregating data...")
    # Aggregate using groupby
    aggregated = sales_df.groupby(
        ['product_store_key', 'id_product', 'id_tienda', 'date']
    ).agg(
        sales_count=('amount', 'count'),
        avg_amount=('amount', 'mean'),
        amount_std=('amount', 'std'),
        min_amount=('amount', 'min'),
        max_amount=('amount', 'max')
    ).reset_index()
    
    # Fill NaN std with 0
    aggregated['amount_std'] = aggregated['amount_std'].fillna(0)
    
    # Add product name
    aggregated['product_name'] = aggregated['id_product'].map(product_names)
    
    print("Creating complete time series...")
    
    # Create complete date range
    min_date = aggregated['date'].min()
    max_date = aggregated['date'].max()
    all_dates = pd.date_range(min_date, max_date, freq=frequency)
    
    result_list = []
    
    for (product_store_key, id_product, id_tienda), group in aggregated.groupby(['product_store_key', 'id_product', 'id_tienda']):
        # Get product name from the group
        product_name_val = group['product_name'].iloc[0]
        
        # Set date as index and reindex to all dates
        group_indexed = group.set_index('date')
        group_reindexed = group_indexed.reindex(all_dates)
        
        # Reset index to get date column back
        group_reindexed = group_reindexed.reset_index()
        
        # Rename 'index' or 'date' column to 'time_sale'
        if 'index' in group_reindexed.columns:
            group_reindexed = group_reindexed.rename(columns={'index': 'time_sale'})
        else:
            group_reindexed = group_reindexed.rename(columns={'date': 'time_sale'})
        
        # Add back identifier columns
        group_reindexed['product_store_key'] = product_store_key
        group_reindexed['id_product'] = id_product
        group_reindexed['id_tienda'] = id_tienda
        group_reindexed['product_name'] = product_name_val
        
        # Fill missing values
        group_reindexed['sales_count'] = group_reindexed['sales_count'].fillna(0)
        
        # Fill numeric columns with 0 (simpler and faster)
        for col in ['avg_amount', 'amount_std', 'min_amount', 'max_amount']:
            if col in group_reindexed.columns:
                group_reindexed[col] = group_reindexed[col].fillna(0)
        
        result_list.append(group_reindexed)
    
    # Combine all results
    result_df = pd.concat(result_list, ignore_index=True)
    
    # Ensure time_sale is the correct type and order
    if 'time_sale' not in result_df.columns:
        raise ValueError("Failed to create 'time_sale' column")
    
    # Convert to datetime and then to Python datetime for Pydantic
    result_df['time_sale'] = pd.to_datetime(result_df['time_sale'])
    
    # Sort by time_sale for each group (optional but recommended)
    result_df = result_df.sort_values(['product_store_key', 'time_sale']).reset_index(drop=True)
    
    # CRITICAL: Convert to Python datetime for Pydantic compatibility
    result_df['time_sale'] = result_df['time_sale'].dt.to_pydatetime()
    
    # Verify columns
    print(f"✓ Final columns: {result_df.columns.tolist()}")
    print(f"✓ Sample time_sale: {result_df['time_sale'].iloc[0] if len(result_df) > 0 else 'N/A'}")
    print(f"✓ time_sale type: {type(result_df['time_sale'].iloc[0]) if len(result_df) > 0 else 'N/A'}")
    
    return result_df


def extract_product_store_series_optimized(
    aggregated_data: pd.DataFrame, 
    value_column: str = 'sales_count'
) -> List[ProductSeries]:
    """
    Extract time series for each product-store combination.
    """
    # Verify required columns exist
    required_cols = ['product_store_key', 'id_product', 'id_tienda', 'time_sale', value_column, 'product_name']
    for col in required_cols:
        if col not in aggregated_data.columns:
            raise KeyError(f"Required column '{col}' not found in aggregated_data")
    
    product_series_list = []
    
    for (product_store_key, id_product, id_tienda), group in aggregated_data.groupby(['product_store_key', 'id_product', 'id_tienda']):
        # Sort by time_sale
        group_sorted = group.sort_values('time_sale')
        
        # Extract series and dates as NUMPY ARRAYS (not lists)
        series = group_sorted[value_column].values.astype(float)  # Convert to numpy float array
        dates = group_sorted['time_sale'].values  # Keep as numpy array
        
        # Convert dates to datetime if needed
        if len(dates) > 0:
            # Ensure dates are datetime64
            dates = pd.to_datetime(dates).values
        
        # Minimum length requirement (at least 7 observations)
        if len(series) >= 7:
            # Create descriptive name
            product_name = group_sorted['product_name'].iloc[0]
            descriptive_name = f"{product_name} (Store {id_tienda})"
            
            # Convert start/end dates to Python datetime
            start_date = dates[0]
            end_date = dates[-1]
            
            # Convert to Python datetime if needed
            if hasattr(start_date, 'to_pydatetime'):
                start_date = start_date.to_pydatetime()
            if hasattr(end_date, 'to_pydatetime'):
                end_date = end_date.to_pydatetime()
            
            product_series_list.append(ProductSeries(
                product_id=f"{product_store_key}",
                product_name=descriptive_name,
                series=series,  # Keep as numpy array
                dates=dates,    # Keep as numpy array
                start_date=start_date,
                end_date=end_date
            ))
    
    print(f"✓ Extracted {len(product_series_list)} product-store time series")
    return product_series_list


def extract_product_store_series_optimized(
    aggregated_data: pd.DataFrame, 
    value_column: str = 'sales_count'
) -> List[ProductSeries]:
    """
    Extract time series for each product-store combination.
    
    Parameters:
    -----------
    aggregated_data : pd.DataFrame
        Output from aggregate_sales_by_product_store_optimized
    value_column : str
        Column to use as the series values (default: 'sales_count')
    
    Returns:
    --------
    List[ProductSeries]
        List of ProductSeries objects, each representing a product-store time series
    """
    product_series_list = []
    
    for (product_store_key, id_product, id_tienda), group in aggregated_data.groupby(['product_store_key', 'id_product', 'id_tienda']):
        group_sorted = group.sort_values('time_sale')
        series = group_sorted[value_column].values
        dates = group_sorted['time_sale'].values
        
        # Minimum length requirement (at least 7 observations)
        if len(series) >= 7:
            # Create descriptive name that includes store info
            product_name = group_sorted['product_name'].iloc[0]
            descriptive_name = f"{product_name} (Store {id_tienda})"
            
            product_series_list.append(ProductSeries(
                product_id=f"{product_store_key}",  # Unique key combining product and store
                product_name=descriptive_name,
                series=series,
                dates=dates,
                start_date=dates[0],
                end_date=dates[-1]
            ))
    
    return product_series_list


# ============================================
# FUNCIÓN PRINCIPAL PARA CREAR EL MULTIGRAFO
# ============================================

def create_product_store_multigraph(
    sales_df: pd.DataFrame,
    product_analyses: List[AnalysisResult],  # Resultados de analyze_all_products_optimized
    frequency: str = 'D',
    stationarity_threshold: float = 0.05,  # p-value threshold for stationarity
    trend_similarity_threshold: float = 0.1,  # Slope difference threshold
    min_purchases_together: int = 3,  # Minimum times products bought together
    include_store_nodes: bool = True,
    include_product_nodes: bool = True
) -> nx.MultiGraph:
    """
    Crea un multigrafo con productos y tiendas como nodos.
    
    Tipos de relaciones (edges):
    1. PRODUCT_SIMILARITY: Entre productos con misma estacionariedad y tendencia similar
    2. SAME_STORE: Entre productos que se venden en la misma tienda
    3. PURCHASED_TOGETHER: Entre productos comprados juntos en el mismo ticket/transacción
    
    Parámetros:
    -----------
    sales_df : pd.DataFrame
        DataFrame con ventas (debe tener columnas: id_product, id_tienda, type, date_minuts, amount)
    product_analyses : List[AnalysisResult]
        Resultados del análisis de estacionariedad y tendencia para cada producto
    frequency : str
        Frecuencia temporal para análisis
    stationarity_threshold : float
        Umbral de p-value para considerar una serie estacionaria
    trend_similarity_threshold : float
        Diferencia máxima en pendiente para considerar tendencias similares
    min_purchases_together : int
        Número mínimo de compras conjuntas para crear una relación
    include_store_nodes : bool
        Incluir tiendas como nodos
    include_product_nodes : bool
        Incluir productos como nodos
    
    Returns:
    --------
    nx.MultiGraph
        Multigrafo con todos los nodos y relaciones
    """
    
    G = nx.MultiGraph()
    
    # 1. Agregar nodos de productos con sus propiedades
    if include_product_nodes:
        product_properties = {}
        for analysis in product_analyses:
            product_id = analysis.product_id
            
            # Determinar estacionariedad basado en p-value
            is_stationary = analysis.stationarity['p_value'] < stationarity_threshold
            
            # Determinar tipo de tendencia
            trend_direction = analysis.trend['trend_direction']
            slope = analysis.trend['slope']
            
            # Guardar propiedades para el nodo
            product_properties[product_id] = {
                'node_type': 'product',
                'product_name': analysis.product_name,
                'is_stationary': is_stationary,
                'stationarity_pvalue': analysis.stationarity['p_value'],
                'trend_direction': trend_direction,
                'trend_slope': slope,
                'trend_magnitude': analysis.trend['magnitude_per_period'],
                'mean_sales': analysis.statistical_features['mean'],
                'cv': analysis.statistical_features['coefficient_variation'],
                'main_period': analysis.periodicity['main_period'] if analysis.periodicity['main_period'] else 0
            }
            
            G.add_node(product_id, **product_properties[product_id])
    
    # 2. Agregar nodos de tiendas
    if include_store_nodes and 'id_tienda' in sales_df.columns:
        stores = sales_df['id_tienda'].unique()
        for store_id in stores:
            G.add_node(str(store_id), node_type='store', store_id=store_id)
    
    # 3. RELACIÓN 1: PRODUCT_SIMILARITY - Productos con misma estacionariedad y tendencia similar
    if include_product_nodes and len(product_analyses) > 1:
        for i, analysis1 in enumerate(product_analyses):
            for analysis2 in product_analyses[i+1:]:
                # Verificar misma estacionariedad
                pvalue1 = analysis1.stationarity['p_value']
                pvalue2 = analysis2.stationarity['p_value']
                same_stationarity = (pvalue1 < stationarity_threshold) == (pvalue2 < stationarity_threshold)
                
                # Verificar tendencia similar
                slope1 = analysis1.trend['slope']
                slope2 = analysis2.trend['slope']
                similar_trend = abs(slope1 - slope2) < trend_similarity_threshold
                
                # Verificar misma dirección de tendencia
                same_direction = analysis1.trend['trend_direction'] == analysis2.trend['trend_direction']
                
                if same_stationarity and similar_trend and same_direction:
                    # Calcular similitud (peso del edge)
                    similarity_score = _calculate_product_similarity(analysis1, analysis2)
                    
                    G.add_edge(
                        analysis1.product_id, 
                        analysis2.product_id,
                        relation_type='PRODUCT_SIMILARITY',
                        weight=similarity_score,
                        same_stationarity=same_stationarity,
                        slope_diff=abs(slope1 - slope2),
                        same_trend_direction=same_direction
                    )
    
    # 4. RELACIÓN 2: SAME_STORE - Productos que se venden en la misma tienda
    if include_product_nodes and include_store_nodes and 'id_tienda' in sales_df.columns:
        # Crear mapping de qué productos se venden en cada tienda
        store_products = defaultdict(set)
        
        # Asumiendo que sales_df tiene las transacciones
        product_store_map = sales_df[['id_product', 'id_tienda']].drop_duplicates()
        
        for _, row in product_store_map.iterrows():
            product_id = str(row['id_product'])
            store_id = str(row['id_tienda'])
            store_products[store_id].add(product_id)
        
        # Conectar productos con tiendas
        for store_id, products in store_products.items():
            if store_id in G:
                for product_id in products:
                    if product_id in G:
                        # Calcular frecuencia de venta en esta tienda
                        sales_in_store = sales_df[
                            (sales_df['id_product'] == int(product_id)) & 
                            (sales_df['id_tienda'] == int(store_id))
                        ].shape[0]
                        
                        G.add_edge(
                            product_id, 
                            store_id,
                            relation_type='SAME_STORE',
                            weight=min(1.0, sales_in_store / 100),  # Normalizado
                            sales_count=sales_in_store
                        )
    
    # 5. RELACIÓN 3: PURCHASED_TOGETHER - Productos comprados juntos
    if include_product_nodes and 'type' in sales_df.columns:
        # Asumiendo que 'type' puede ser el número de ticket/transacción
        # Si no existe, podemos agrupar por fecha y tienda como proxy
        
        # Método 1: Si existe columna de transacción/ticket
        if 'ticket_id' in sales_df.columns:
            transaction_groups = sales_df.groupby('ticket_id')['id_product'].apply(set)
        else:
            # Método 2: Agrupar por tienda, fecha y hora (transacciones en mismo minuto)
            sales_df['transaction_key'] = sales_df['id_tienda'].astype(str) + '_' + sales_df['date_minuts'].astype(str)
            transaction_groups = sales_df.groupby('transaction_key')['id_product'].apply(set)
        
        # Contar frecuencias de pares de productos comprados juntos
        pair_counts = defaultdict(int)
        
        for products_set in transaction_groups:
            if len(products_set) > 1:
                for p1, p2 in combinations(products_set, 2):
                    key = tuple(sorted([str(p1), str(p2)]))
                    pair_counts[key] += 1
        
        # Crear edges para pares que superan el umbral
        for (p1, p2), count in pair_counts.items():
            if count >= min_purchases_together and p1 in G and p2 in G:
                # Calcular fuerza de la relación
                total_purchases_p1 = sales_df[sales_df['id_product'] == int(p1)].shape[0]
                total_purchases_p2 = sales_df[sales_df['id_product'] == int(p2)].shape[0]
                
                # Lift: qué tan probable es que se compren juntos vs individualmente
                expected = (total_purchases_p1 * total_purchases_p2) / len(transaction_groups)
                lift = count / (expected + 1e-8)
                
                G.add_edge(
                    p1, p2,
                    relation_type='PURCHASED_TOGETHER',
                    weight=min(1.0, count / min_purchases_together),
                    times_together=count,
                    lift=lift,
                    confidence=count / min(total_purchases_p1, total_purchases_p2)
                )
    
    return G


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def _calculate_product_similarity(analysis1: AnalysisResult, analysis2: AnalysisResult) -> float:
    """
    Calcula una puntuación de similitud entre dos productos basada en sus características
    
    Returns:
    --------
    float: Puntuación entre 0 y 1
    """
    scores = []
    
    # Similitud de pendiente de tendencia
    slope_sim = 1 - min(1.0, abs(analysis1.trend['slope'] - analysis2.trend['slope']) / 
                       (abs(analysis1.trend['slope']) + abs(analysis2.trend['slope']) + 1e-8))
    scores.append(slope_sim * 0.3)
    
    # Similitud de coeficiente de variación (variabilidad)
    cv1 = analysis1.statistical_features['coefficient_variation']
    cv2 = analysis2.statistical_features['coefficient_variation']
    cv_sim = 1 - min(1.0, abs(cv1 - cv2) / (max(cv1, cv2) + 1e-8))
    scores.append(cv_sim * 0.2)
    
    # Similitud de volumen de ventas (media normalizada)
    mean1 = analysis1.statistical_features['mean']
    mean2 = analysis2.statistical_features['mean']
    mean_sim = min(mean1, mean2) / (max(mean1, mean2) + 1e-8)
    scores.append(mean_sim * 0.2)
    
    # Similitud de estacionariedad
    stationary_sim = 1.0 if analysis1.stationarity['is_stationary'] == analysis2.stationarity['is_stationary'] else 0.0
    scores.append(stationary_sim * 0.15)
    
    # Similitud de periodicidad
    period1 = analysis1.periodicity['main_period'] or 0
    period2 = analysis2.periodicity['main_period'] or 0
    period_sim = 1.0 if period1 == period2 else 0.0
    scores.append(period_sim * 0.15)
    
    return sum(scores)


def _get_transaction_id(row: pd.Series) -> str:
    """
    Genera un ID de transacción basado en tienda y timestamp cercano
    """
    # Si hay diferencia de menos de 5 minutos, consideramos misma transacción
    return f"{row['id_tienda']}_{row['date_minuts'].floor('5min')}"


# ============================================
# FUNCIONES DE ANÁLISIS DEL GRAFO
# ============================================

def analyze_multigraph(G: nx.MultiGraph) -> Dict:
    """
    Analiza el multigrafo y extrae métricas importantes
    """
    results = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_products': sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'product'),
        'n_stores': sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'store'),
        'degree_centrality': nx.degree_centrality(G),
        'clustering_coefficient': nx.clustering(G),
        'connected_components': list(nx.connected_components(G)),
        'n_connected_components': nx.number_connected_components(G)
    }
    
    # Análisis por tipo de relación
    relation_counts = defaultdict(int)
    for u, v, data in G.edges(data=True):
        relation_counts[data['relation_type']] += 1
    results['relation_counts'] = dict(relation_counts)
    
    # Productos más conectados
    if results['n_products'] > 0:
        product_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'product']
        if product_nodes:
            degrees = [(n, G.degree(n)) for n in product_nodes]
            results['most_connected_products'] = sorted(degrees, key=lambda x: x[1], reverse=True)[:10]
    
    return results


def analyze_graph_advanced(G: nx.MultiGraph) -> Dict:
    """
    Análisis avanzado del multigrafo
    """
    results = {
        'basic_stats': {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_products': sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'product'),
            'n_stores': sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'store'),
            'density': nx.density(G) if G.number_of_nodes() > 1 else 0,
            'is_connected': nx.is_connected(G) if G.number_of_nodes() > 0 else False,
            'n_connected_components': nx.number_connected_components(G),
        },
        'relation_analysis': {},
        'product_analysis': {},
        'store_analysis': {},
        'centrality_analysis': {},
        'community_analysis': {}
    }
    
    # Análisis por tipo de relación
    relation_counts = defaultdict(int)
    relation_weights = defaultdict(list)
    
    for u, v, data in G.edges(data=True):
        rel_type = data.get('relation_type', 'unknown')
        relation_counts[rel_type] += 1
        relation_weights[rel_type].append(data.get('weight', 0))
    
    results['relation_analysis'] = {
        'counts': dict(relation_counts),
        'avg_weights': {k: float(np.mean(v)) for k, v in relation_weights.items() if v},
        'max_weights': {k: float(np.max(v)) for k, v in relation_weights.items() if v}
    }
    
    # Análisis de productos
    product_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'product']
    
    if product_nodes:
        # Productos por estacionariedad
        stationary_products = [n for n in product_nodes if G.nodes[n].get('is_stationary', False)]
        non_stationary_products = [n for n in product_nodes if not G.nodes[n].get('is_stationary', False)]
        
        # Productos por tendencia
        trend_counts = defaultdict(int)
        for n in product_nodes:
            trend = G.nodes[n].get('trend_direction', 'unknown')
            trend_counts[trend] += 1
        
        # Productos más conectados (grado)
        product_degrees = [(n, G.degree(n)) for n in product_nodes]
        most_connected = sorted(product_degrees, key=lambda x: x[1], reverse=True)[:10]
        
        results['product_analysis'] = {
            'total_products': len(product_nodes),
            'stationary_count': len(stationary_products),
            'non_stationary_count': len(non_stationary_products),
            'stationary_percentage': len(stationary_products) / len(product_nodes) * 100 if product_nodes else 0,
            'trend_distribution': dict(trend_counts),
            'most_connected_products': [
                {'product_id': p, 'degree': d, 'name': G.nodes[p].get('product_name', p)} 
                for p, d in most_connected
            ],
            'avg_degree': float(np.mean([d for _, d in product_degrees])) if product_degrees else 0,
            'max_degree': max([d for _, d in product_degrees]) if product_degrees else 0
        }
    
    # Análisis de tiendas
    store_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'store']
    
    if store_nodes:
        # Tiendas con más productos
        store_product_counts = []
        for store in store_nodes:
            n_products = sum(1 for neighbor in G.neighbors(store) 
                           if G.nodes[neighbor].get('node_type') == 'product')
            store_product_counts.append((store, n_products))
        
        top_stores = sorted(store_product_counts, key=lambda x: x[1], reverse=True)[:10]
        
        results['store_analysis'] = {
            'total_stores': len(store_nodes),
            'avg_products_per_store': float(np.mean([c for _, c in store_product_counts])) if store_product_counts else 0,
            'top_stores': [
                {'store_id': s, 'product_count': c} 
                for s, c in top_stores
            ]
        }
    
    # Análisis de centralidad (solo para componentes conexas grandes)
    if G.number_of_nodes() > 1:
        # Obtener la componente más grande
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        G_largest = G.subgraph(largest_component)
        
        if len(largest_component) > 2:
            results['centrality_analysis'] = {
                'degree_centrality_top': [
                    {'node': n, 'centrality': c}
                    for n, c in sorted(nx.degree_centrality(G_largest).items(), 
                                      key=lambda x: x[1], reverse=True)[:10]
                ],
                'betweenness_centrality_top': [
                    {'node': n, 'centrality': c}
                    for n, c in sorted(nx.betweenness_centrality(G_largest).items(),
                                      key=lambda x: x[1], reverse=True)[:5]
                ],
                'closeness_centrality_top': [
                    {'node': n, 'centrality': c}
                    for n, c in sorted(nx.closeness_centrality(G_largest).items(),
                                      key=lambda x: x[1], reverse=True)[:5]
                ]
            }
    
    # Análisis de comunidades (clusterización)
    if len(product_nodes) > 10:
        # Usar solo nodos producto para comunidades
        G_products = G.subgraph(product_nodes)
        if G_products.number_of_edges() > 0:
            try:
                from networkx.algorithms import community
                # Detectar comunidades usando greedy modularity
                communities = community.greedy_modularity_communities(G_products)
                
                results['community_analysis'] = {
                    'n_communities': len(communities),
                    'modularity': community.modularity(G_products, communities),
                    'community_sizes': [len(c) for c in communities],
                    'largest_communities': [
                        {
                            'size': len(comm),
                            'products': [G.nodes[p].get('product_name', p) for p in list(comm)[:10]]
                        }
                        for comm in sorted(communities, key=len, reverse=True)[:5]
                    ]
                }
            except Exception as e:
                logger.warning(f"Community detection failed: {e}")
                results['community_analysis'] = {'error': str(e)}
    
    return results


def find_product_clusters_by_stationarity(G: nx.MultiGraph) -> Dict[str, List[str]]:
    """
    Encuentra clusters de productos según su estacionariedad
    """
    clusters = {
        'stationary': [],
        'non_stationary': []
    }
    
    for node, attrs in G.nodes(data=True):
        if attrs.get('node_type') == 'product':
            if attrs.get('is_stationary', False):
                clusters['stationary'].append(node)
            else:
                clusters['non_stationary'].append(node)
    
    return clusters


def find_products_with_similar_trend(G: nx.MultiGraph, product_id: str, min_similarity: float = 0.7) -> List[str]:
    """
    Encuentra productos con tendencia similar a uno dado
    """
    similar_products = []
    
    if product_id not in G:
        return similar_products
    
    for neighbor in G.neighbors(product_id):
        edge_data = G.get_edge_data(product_id, neighbor)
        for key, data in edge_data.items():
            if data.get('relation_type') == 'PRODUCT_SIMILARITY':
                if data.get('weight', 0) >= min_similarity:
                    similar_products.append(neighbor)
    
    return similar_products


def find_similar_products_in_graph(G: nx.MultiGraph, product_id: str, min_weight: float = 0.5) -> List[Dict]:
    """
    Encuentra productos similares basado en relaciones del grafo
    """
    similar = []
    
    if product_id not in G:
        return similar
    
    for neighbor in G.neighbors(product_id):
        if G.nodes[neighbor].get('node_type') == 'product':
            # Obtener todas las relaciones entre estos productos
            edges_data = G.get_edge_data(product_id, neighbor)
            max_weight = 0
            relation_types = []
            
            for key, data in edges_data.items():
                weight = data.get('weight', 0)
                if weight > max_weight:
                    max_weight = weight
                relation_types.append(data.get('relation_type', 'unknown'))
            
            if max_weight >= min_weight:
                similar.append({
                    'product_id': neighbor,
                    'product_name': G.nodes[neighbor].get('product_name', neighbor),
                    'similarity_score': max_weight,
                    'relation_types': list(set(relation_types)),
                    'is_stationary': G.nodes[neighbor].get('is_stationary', False),
                    'trend_direction': G.nodes[neighbor].get('trend_direction', 'unknown')
                })
    
    # Ordenar por similitud
    similar.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similar

# ============================================
# FUNCIONES PARA SERIALIZAR EL GRAFO
# ============================================

def serialize_graph_to_json(G: nx.MultiGraph) -> Dict:
    """
    Serializa el multigrafo a un formato JSON serializable
    """
    # Nodos
    nodes = []
    for node, attrs in G.nodes(data=True):
        node_data = {
            'id': str(node),
            'type': attrs.get('node_type', 'unknown'),
            'properties': {}
        }
        
        # Incluir propiedades relevantes
        if attrs.get('node_type') == 'product':
            node_data['properties'] = {
                'product_name': attrs.get('product_name', str(node)),
                'is_stationary': attrs.get('is_stationary', False),
                'stationarity_pvalue': attrs.get('stationarity_pvalue', None),
                'trend_direction': attrs.get('trend_direction', 'unknown'),
                'trend_slope': attrs.get('trend_slope', 0),
                'trend_magnitude': attrs.get('trend_magnitude', 0),
                'mean_sales': attrs.get('mean_sales', 0),
                'cv': attrs.get('cv', 0),
                'main_period': attrs.get('main_period', 0)
            }
        elif attrs.get('node_type') == 'store':
            node_data['properties'] = {
                'store_id': attrs.get('store_id', str(node))
            }
        
        nodes.append(node_data)
    
    # Aristas
    edges = []
    for u, v, key, data in G.edges(data=True, keys=True):
        edge_data = {
            'source': str(u),
            'target': str(v),
            'relation_type': data.get('relation_type', 'unknown'),
            'weight': data.get('weight', 1.0),
            'properties': {}
        }
        
        # Incluir propiedades específicas según tipo de relación
        if data.get('relation_type') == 'PRODUCT_SIMILARITY':
            edge_data['properties'] = {
                'same_stationarity': data.get('same_stationarity', False),
                'slope_diff': data.get('slope_diff', 0),
                'same_trend_direction': data.get('same_trend_direction', False)
            }
        elif data.get('relation_type') == 'SAME_STORE':
            edge_data['properties'] = {
                'sales_count': data.get('sales_count', 0)
            }
        elif data.get('relation_type') == 'PURCHASED_TOGETHER':
            edge_data['properties'] = {
                'times_together': data.get('times_together', 0),
                'lift': data.get('lift', 0),
                'confidence': data.get('confidence', 0)
            }
        
        edges.append(edge_data)
    
    return {
        'nodes': nodes,
        'edges': edges,
        'summary': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'product_nodes': sum(1 for n in nodes if n['type'] == 'product'),
            'store_nodes': sum(1 for n in nodes if n['type'] == 'store')
        }
    }


# ============================================
# PIPELINES INTEGRADOS DE PROCESAMIENTO
# ============================================

# Modified version of run_cluster_analysis_pipeline for product-store analysis
def run_product_store_cluster_analysis(
    df: pd.DataFrame,
    frequency: str = 'D',
    value_column: str = 'sales_count',
    dbscan_eps: float = 1.0,
    dbscan_min_samples: int = 3,
    progress_callback=None
) -> Dict:
    """
    Complete clustering analysis pipeline for product-store time series.
    
    This analyzes each product's behavior in each store independently.
    """
    # 1. Aggregate data by product-store
    aggregated_data = aggregate_sales_by_product_store_optimized(df, frequency)
    
    # 2. Extract time series for each product-store combination
    product_series_list = extract_product_store_series_optimized(aggregated_data, value_column)
    
    print(f"Generated {len(product_series_list)} product-store time series")
    
    # 3. Analyze all product-store series
    analyses_results = analyze_all_products_optimized(product_series_list, progress_callback=progress_callback)
    
    # 4. Create feature matrix
    feature_matrix = create_feature_matrix_optimized(analyses_results)
    
    # Add store information to feature matrix
    # Extract store_id from product_id (which contains product_store_key)
    feature_matrix['store_id'] = feature_matrix['product_id'].apply(lambda x: x.split('_')[1] if '_' in x else 'unknown')
    feature_matrix['product_id_original'] = feature_matrix['product_id'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    
    # 5. Normalize features
    features_normalized, scaler, feature_cols = normalize_features_optimized(feature_matrix)
    
    # 6. Perform clustering
    kmeans_results = perform_kmeans_optimized(features_normalized)
    hierarchical_results = perform_hierarchical_optimized(features_normalized, n_clusters=kmeans_results['optimal_k'])
    dbscan_results = perform_dbscan_optimized(features_normalized, eps=dbscan_eps, min_samples=dbscan_min_samples)
    
    # 7. Extract cluster profiles
    cluster_profiles = extract_cluster_profiles_optimized(feature_matrix, kmeans_results['labels'])
    
    # 8. Create cluster summary
    cluster_summary = []
    for cluster_id, profile in cluster_profiles.items():
        cluster_summary.append({
            'cluster_id': profile['cluster_id'],
            'cluster_name': profile['cluster_name'],
            'n_products': profile['n_products'],
            'percentage': profile['percentage'],
            'characteristics': ', '.join(profile['characteristics'])
        })
    
    return {
        'analyses_results': [asdict(r) for r in analyses_results],
        'feature_matrix': feature_matrix,
        'kmeans_results': kmeans_results,
        'hierarchical_results': hierarchical_results,
        'dbscan_results': dbscan_results,
        'cluster_summary': cluster_summary,
        'cluster_profiles': cluster_profiles
    }

# Helper function to analyze a specific product across all stores
def analyze_product_across_stores(
    df: pd.DataFrame,
    product_id: int,
    frequency: str = 'D',
    value_column: str = 'sales_count'
) -> pd.DataFrame:
    """
    Analyze how a specific product behaves across different stores.
    Returns a DataFrame with store-level time series features.
    """
    aggregated_data = aggregate_sales_by_product_store_optimized(df, frequency)
    
    # Filter for specific product
    product_data = aggregated_data[aggregated_data['id_product'] == product_id]
    
    if product_data.empty:
        return pd.DataFrame()
    
    # Extract series for each store
    product_series_list = extract_product_store_series_optimized(product_data, value_column)
    
    # Analyze each store's series
    results = []
    for ps in product_series_list:
        analysis = analyze_product_optimized(ps)
        if analysis:
            store_id = ps.product_id.split('_')[1] if '_' in ps.product_id else 'unknown'
            result_dict = asdict(analysis)
            result_dict['store_id'] = store_id
            result_dict['raw_series'] = ps.series.tolist()  # Include raw series for this specific analysis
            results.append(result_dict)
    
    return pd.DataFrame(results)


def run_complete_multigraph_pipeline(
    sales_df: pd.DataFrame,
    frequency: str = 'D',
    stationarity_threshold: float = 0.05,
    min_purchases_together: int = 3
) -> Dict:
    """
    Pipeline completo: análisis + creación del multigrafo
    """
    from datetime import datetime
    
    print("1. Agregando datos por tiempo...")
    aggregated_data = aggregate_sales_by_time_optimized(sales_df, frequency)
    
    print("2. Extrayendo series de productos...")
    product_series_list = extract_product_series_optimized(aggregated_data)
    
    print("3. Analizando estacionariedad y tendencia de productos...")
    analyses_results = analyze_all_products_optimized(product_series_list)
    
    print(f"   Analizados {len(analyses_results)} productos")
    
    print("4. Creando multigrafo...")
    G = create_product_store_multigraph(
        sales_df=sales_df,
        product_analyses=analyses_results,
        frequency=frequency,
        stationarity_threshold=stationarity_threshold,
        min_purchases_together=min_purchases_together
    )
    
    print("5. Analizando estructura del grafo...")
    graph_analysis = analyze_multigraph(G)
    
    print("6. Identificando clusters por estacionariedad...")
    stationarity_clusters = find_product_clusters_by_stationarity(G)
    
    return {
        'graph': G,
        'product_analyses': analyses_results,
        'graph_analysis': graph_analysis,
        'stationarity_clusters': stationarity_clusters
    }


def filter_constant_series(product_series_list: List[ProductSeries]) -> Tuple[List[ProductSeries], List[str]]:
    """Filter out constant series"""
    valid_series = []
    constant_products = []
    
    for ps in product_series_list:
        if len(np.unique(ps.series)) == 1 or np.std(ps.series) == 0:
            constant_products.append(ps.product_name)
            logger.warning(f"Product {ps.product_name} has constant series, excluded")
        else:
            valid_series.append(ps)
    
    return valid_series, constant_products
