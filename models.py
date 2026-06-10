# models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import numpy as np 


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class StationarityResult(BaseModel):
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_stationary: bool
    used_lag: int


class TrendResult(BaseModel):
    slope: float
    intercept: float
    r_squared: float
    p_value_linear: float
    kendall_tau: float
    p_value_kendall: float
    trend_direction: TrendDirection
    magnitude_per_period: float


class PeriodicityResult(BaseModel):
    significant_periods: List[int]
    main_period: Optional[int]


class StatisticalFeatures(BaseModel):
    mean: float
    median: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    min: float
    max: float
    range: float
    q25: float
    q50: float
    q75: float
    iqr: float
    coefficient_variation: float
    zero_crossings: float
    abs_energy: float


class AnomalyResult(BaseModel):
    anomaly_indices: List[int]
    anomaly_count: float
    anomaly_ratio: float


class DateRange(BaseModel):
    start: datetime
    end: datetime


class ProductTimeSeriesAnalysis(BaseModel):
    product_id: str
    product_name: str
    series_length: int
    date_range: DateRange
    stationarity: StationarityResult
    trend: TrendResult
    periodicity: PeriodicityResult
    statistical_features: StatisticalFeatures
    anomalies: AnomalyResult


class ProductTimeSeriesSummary(BaseModel):
    product_id: str
    product_name: str
    series_length: int
    date_range: DateRange
    is_stationary: bool
    critical_values: Dict[str, float]
    trend_direction: TrendDirection
    significant_periods: List[int]


class ClusterProfile(BaseModel):
    cluster_id: int
    cluster_name: str
    n_products: int
    percentage: float
    products: List[str]
    product_ids: List[str]
    characteristics: List[str]


class KMeansResults(BaseModel):
    optimal_k: int
    silhouette_scores: Dict[int, float]
    inertia_values: Dict[int, float]


class HierarchicalResults(BaseModel):
    n_clusters: int
    silhouette_score: float


class DBSCANResults(BaseModel):
    n_clusters: int
    n_noise: int
    silhouette_score: Optional[float]
    eps: float
    min_samples: int

class ClusterAnalysisResponse(BaseModel):
    analyses_results: List[ProductTimeSeriesAnalysis]
    kmeans_results: KMeansResults
    hierarchical_results: HierarchicalResults
    dbscan_results: DBSCANResults
    cluster_summary: List[ClusterProfile]

class TimeSeriesExtractResponse(BaseModel):
    time_series_results: List[ProductTimeSeriesSummary]

@dataclass
class ProductSeries:
    product_id: str
    product_name: str
    series: np.ndarray  # Debe ser numpy array, NO lista
    dates: np.ndarray   # Debe ser numpy array, NO lista
    start_date: datetime
    end_date: datetime


# ============================================
# MODELOS DE RESPUESTA PARA SWAGGER
# ============================================

class GraphNodeResponse(BaseModel):
    node_id: str
    node_type: str
    properties: Dict[str, Any]

class GraphEdgeResponse(BaseModel):
    source: str
    target: str
    relation_type: str
    weight: float
    properties: Dict[str, Any]

class GraphAnalysisResponse(BaseModel):
    n_nodes: int
    n_edges: int
    n_products: int
    n_stores: int
    n_connected_components: int
    relation_counts: Dict[str, int]
    clustering_coefficient: float
    density: float