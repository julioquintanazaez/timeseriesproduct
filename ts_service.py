# ts_service.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
import pywt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from functools import lru_cache

from datahandle import numpy_to_json_serializable, convert_numpy_nativo

import logging
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. DATA CLASSES FOR TYPE SAFETY
# ============================================

@dataclass
class ProductSeries:
    """Lightweight container for product time series data"""
    product_id: str
    product_name: str
    series: np.ndarray
    dates: np.ndarray
    start_date: datetime
    end_date: datetime
    
    @property
    def length(self) -> int:
        return len(self.series)
    
    def to_summary_dict(self) -> Dict:
        return {
            'product_id': self.product_id,
            'product_name': self.product_name,
            'series_length': self.length,
            'start_date': self.start_date,
            'end_date': self.end_date
        }


@dataclass
class AnalysisResult:
    """Container for analysis results without raw series"""
    product_id: str
    product_name: str
    series_length: int
    date_range: Dict
    stationarity: Dict
    trend: Dict
    periodicity: Dict
    statistical_features: Dict
    anomalies: Dict


# ============================================
# 2. OPTIMIZED DATA PREPROCESSING
# ============================================

def aggregate_sales_by_time_optimized(sales_df: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
    """
    Optimized aggregation using pivot_table for better performance
    """
    sales_df = sales_df.copy()
    sales_df['time_sale'] = pd.to_datetime(sales_df['time_sale'])
    sales_df['date'] = sales_df['time_sale'].dt.floor(frequency)
    
    # Single pass aggregation using pivot_table
    pivot = sales_df.pivot_table(
        index=['id_product', 'date'],
        values='amount',
        aggfunc=['count', 'mean', 'std', 'min', 'max']
    )
    
    pivot.columns = ['sales_count', 'avg_amount', 'amount_std', 'min_amount', 'max_amount']
    pivot = pivot.reset_index()
    
    # Add product names
    product_names = sales_df.groupby('id_product')['name'].first()
    pivot['product_name'] = pivot['id_product'].map(product_names)
    
    # Create complete time series for each product - SIMPLER APPROACH
    all_dates = pd.date_range(pivot['date'].min(), pivot['date'].max(), freq=frequency)
    complete_series = []
    
    for product_id in pivot['id_product'].unique():
        # Get product data
        product_data = pivot[pivot['id_product'] == product_id].copy()
        product_name_val = product_data['product_name'].iloc[0]
        
        # Create complete date range
        product_data = product_data.set_index('date').reindex(all_dates).reset_index()
        product_data.rename(columns={'index': 'time_sale'}, inplace=True)
        
        # Add back product info
        product_data['id_product'] = product_id
        product_data['product_name'] = product_name_val
        
        # Fill missing values
        product_data['sales_count'] = product_data['sales_count'].fillna(0)
        
        # Interpolate price fields
        for col in ['avg_amount', 'amount_std', 'min_amount', 'max_amount']:
            if col in product_data.columns:
                product_data[col] = product_data[col].interpolate(method='linear').bfill()
        
        complete_series.append(product_data)
    
    return pd.concat(complete_series, ignore_index=True)

def extract_product_series_optimized(aggregated_data: pd.DataFrame, value_column: str = 'sales_count') -> List[ProductSeries]:
    """
    Extract all product series in one pass
    """
    product_series_list = []
    
    for product_id, group in aggregated_data.groupby('id_product'):
        group_sorted = group.sort_values('time_sale')
        series = group_sorted[value_column].values
        dates = group_sorted['time_sale'].values
        
        if len(series) >= 7:  # Minimum length requirement
            product_series_list.append(ProductSeries(
                product_id=str(product_id),
                product_name=group_sorted['product_name'].iloc[0],
                series=series,
                dates=dates,
                start_date=dates[0],
                end_date=dates[-1]
            ))
    
    return product_series_list


# ============================================
# 3. OPTIMIZED FEATURE CALCULATIONS (CACHED)
# ============================================

@lru_cache(maxsize=1024)
def _cached_adfuller(series_tuple: tuple) -> Dict:
    """Cached ADF test for repeated series"""
    series = np.array(series_tuple)
    result = adfuller(series, autolag='AIC')
    return {
        'adf_statistic': float(result[0]),
        'p_value': float(result[1]),
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05,
        'used_lag': result[2]
    }


def test_stationarity_optimized(series: np.ndarray) -> Dict:
    """Wrapper for cached stationarity test"""
    return _cached_adfuller(tuple(series))


def detect_trend_optimized(series: np.ndarray) -> Dict:
    """Optimized trend detection"""
    x = np.arange(len(series))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
    tau, p_value_kendall = stats.kendalltau(x, series)
    
    trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'p_value_linear': float(p_value),
        'kendall_tau': float(tau),
        'p_value_kendall': float(p_value_kendall),
        'trend_direction': trend_direction,
        'magnitude_per_period': float(abs(slope))
    }


def detect_periodicity_optimized(series: np.ndarray, max_lags: int = 50) -> Dict:
    """Optimized periodicity detection"""
    max_lags = min(max_lags, len(series) // 2)
    if max_lags < 2:
        return {'significant_periods': [], 'main_period': None}
    
    acf_values = acf(series, nlags=max_lags, fft=True)  # fft=True is faster
    peaks, _ = find_peaks(acf_values[1:], height=0.2)
    peaks = peaks + 1
    
    significant_periods = [int(p) for p in peaks if acf_values[p] > 0.3]
    
    return {
        'significant_periods': significant_periods,
        'main_period': significant_periods[0] if significant_periods else None
    }


def calculate_statistical_features_optimized(series: np.ndarray) -> Dict:
    """Vectorized statistical features calculation"""
    mean_val = np.mean(series)
    std_val = np.std(series)
    
    return {
        'mean': float(mean_val),
        'median': float(np.median(series)),
        'std': float(std_val),
        'variance': float(np.var(series)),
        'skewness': float(stats.skew(series)),
        'kurtosis': float(stats.kurtosis(series)),
        'min': float(np.min(series)),
        'max': float(np.max(series)),
        'range': float(np.ptp(series)),
        'q25': float(np.percentile(series, 25)),
        'q50': float(np.percentile(series, 50)),
        'q75': float(np.percentile(series, 75)),
        'iqr': float(np.percentile(series, 75) - np.percentile(series, 25)),
        'coefficient_variation': float(std_val / (mean_val + 1e-8)),
        'zero_crossings': float(np.sum(np.diff(np.signbit(series)))),
        'abs_energy': float(np.sum(series ** 2))
    }


def detect_anomalies_optimized(series: np.ndarray, threshold: float = 3) -> Dict:
    """Optimized anomaly detection"""
    z_scores = np.abs(stats.zscore(series))
    anomalies = z_scores > threshold
    
    return {
        'anomaly_indices': np.where(anomalies)[0].tolist(),
        'anomaly_count': float(np.sum(anomalies)),
        'anomaly_ratio': float(np.sum(anomalies) / len(series)),
    }


# ============================================
# 4. OPTIMIZED PRODUCT ANALYSIS
# ============================================

def analyze_product_optimized(product_series: ProductSeries) -> Optional[AnalysisResult]:
    """
    Comprehensive analysis for a single product without storing raw series
    """
    series = product_series.series
    
    if len(series) < 7:
        return None
    
    return AnalysisResult(
        product_id=product_series.product_id,
        product_name=product_series.product_name,
        series_length=len(series),
        date_range={
            'start': product_series.start_date,
            'end': product_series.end_date
        },
        stationarity=test_stationarity_optimized(series),
        trend=detect_trend_optimized(series),
        periodicity=detect_periodicity_optimized(series),
        statistical_features=calculate_statistical_features_optimized(series),
        anomalies=detect_anomalies_optimized(series)
    )


def analyze_all_products_optimized(
    product_series_list: List[ProductSeries], 
    batch_size: int = 100,
    progress_callback=None
) -> List[AnalysisResult]:
    """
    Batch process all products with progress tracking
    """
    results = []
    total = len(product_series_list)
    
    for i, product_series in enumerate(product_series_list):
        result = analyze_product_optimized(product_series)
        if result:
            results.append(result)
        
        if progress_callback and (i + 1) % batch_size == 0:
            progress_callback(i + 1, total)
    
    return results


def analyze_all_products_optimized_safe(
    product_series_list: List[ProductSeries], 
    progress_callback=None
) -> List[AnalysisResult]:
    """
    Versión segura que maneja series constantes
    """
    analyses_results = []
    
    for i, ps in enumerate(product_series_list):
        try:
            # Verificar serie constante
            if len(np.unique(ps.series)) == 1 or np.std(ps.series) == 0:
                logger.warning(f"Producto {ps.product_name} tiene serie constante, asignando valores por defecto")
                
                # Crear resultado por defecto para series constantes
                from models import StationarityResult, TrendResult, PeriodicityResult, StatisticalFeatures
                
                result = AnalysisResult(
                    product_id=ps.product_id,
                    product_name=ps.product_name,
                    series=ps.series,
                    dates=ps.dates,
                    stationarity=StationarityResult(
                        is_stationary=True,
                        p_value=1.0,
                        statistic=0.0,
                        critical_values={},
                        method='constant_series_fallback'
                    ),
                    trend=TrendResult(
                        slope=0.0,
                        intercept=np.mean(ps.series),
                        trend_direction='stable',
                        magnitude_per_period=0.0,
                        r_squared=1.0,
                        p_value=1.0
                    ),
                    periodicity=PeriodicityResult(
                        main_period=None,
                        period_strength=0.0,
                        fft_frequencies=[],
                        fft_power=[],
                        autocorrelation_peaks=[]
                    ),
                    statistical_features=StatisticalFeatures(
                        mean=np.mean(ps.series),
                        std=np.std(ps.series),
                        min=np.min(ps.series),
                        max=np.max(ps.series),
                        coefficient_variation=0.0,
                        skewness=0.0,
                        kurtosis=0.0,
                        quantiles={
                            '25%': np.percentile(ps.series, 25),
                            '50%': np.percentile(ps.series, 50),
                            '75%': np.percentile(ps.series, 75)
                        }
                    ),
                    change_points=[],
                    quality_metrics={
                        'length': len(ps.series),
                        'missing_ratio': 0.0,
                        'zero_ratio': 0.0,
                        'is_constant': True
                    }
                )
                
                analyses_results.append(result)
                continue
            
            # Análisis normal para series no constantes
            result = analyze_product_optimized(ps)
            if result:
                analyses_results.append(result)
                
        except Exception as e:
            logger.error(f"Error analyzing product {ps.product_name}: {str(e)}")
            continue
        
        if progress_callback:
            progress_callback(i + 1, len(product_series_list))
    
    return analyses_results


def get_product_summary_optimized(product_series: ProductSeries) -> Dict:
    """
    Lightweight summary for a single product
    """
    series = product_series.series
    
    if len(series) < 7:
        return None
    
    stationarity = test_stationarity_optimized(series)
    trend = detect_trend_optimized(series)
    periodicity = detect_periodicity_optimized(series)
    
    return {
        'product_id': product_series.product_id,
        'product_name': product_series.product_name,
        'series_length': len(series),
        'date_range': {
            'start': product_series.start_date,
            'end': product_series.end_date
        },
        'is_stationary': stationarity['is_stationary'],
        'critical_values': stationarity['critical_values'],
        'trend_direction': trend['trend_direction'],
        'significant_periods': periodicity['significant_periods']
    }


# ============================================
# 5. OPTIMIZED FEATURE MATRIX
# ============================================

def create_feature_matrix_optimized(analyses_results: List[AnalysisResult]) -> pd.DataFrame:
    """
    Create feature matrix from analysis results without raw series
    """
    features_list = []
    
    for analysis in analyses_results:
        features = {
            'product_id': analysis.product_id,
            'product_name': analysis.product_name,
            'series_length': analysis.series_length,
            
            # Statistical features
            **{f'stat_{k}': v for k, v in analysis.statistical_features.items()},
            
            # Trend features
            'trend_slope': analysis.trend['slope'],
            'trend_direction': analysis.trend['trend_direction'],
            'trend_magnitude': analysis.trend['magnitude_per_period'],
            'trend_r_squared': analysis.trend['r_squared'],
            
            # Stationarity features
            'stationarity_pvalue': analysis.stationarity['p_value'],
            'is_stationary': analysis.stationarity['is_stationary'],
            'adf_statistic': analysis.stationarity['adf_statistic'],
            
            # Periodicity features
            'main_period': analysis.periodicity['main_period'] or 0,
            'n_significant_periods': len(analysis.periodicity['significant_periods']),
            
            # Anomaly features
            'anomaly_count': analysis.anomalies['anomaly_count'],
            'anomaly_ratio': analysis.anomalies['anomaly_ratio'],
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def normalize_features_optimized(
    feature_matrix: pd.DataFrame, 
    exclude_columns: List[str] = None
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Normalize features for clustering
    """
    if exclude_columns is None:
        exclude_columns = ['product_id', 'product_name', 'is_stationary', 'trend_direction']
    
    feature_cols = [col for col in feature_matrix.columns if col not in exclude_columns]
    
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix[feature_cols])
    
    return normalized_features, scaler, feature_cols


# ============================================
# 6. OPTIMIZED CLUSTERING
# ============================================

def perform_kmeans_optimized(
    features_normalized: np.ndarray, 
    n_clusters_range: range = range(2, 11)
) -> Dict:
    """
    Optimized K-means clustering
    """
    best_k = 2
    best_silhouette = -1
    
    silhouette_scores = {}
    inertia_values = {}
    
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features_normalized)
        
        silhouette = silhouette_score(features_normalized, labels)
        silhouette_scores[k] = silhouette
        inertia_values[k] = kmeans.inertia_
        
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
    
    # Final model
    best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    final_labels = best_kmeans.fit_predict(features_normalized)
    
    return {
        'labels': final_labels.tolist(),
        'optimal_k': best_k,
        'silhouette_scores': silhouette_scores,
        'inertia_values': inertia_values
    }


def perform_hierarchical_optimized(features_normalized: np.ndarray, n_clusters: int = 5) -> Dict:
    """
    Optimized hierarchical clustering
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(features_normalized)
    
    silhouette_avg = silhouette_score(features_normalized, labels)
    
    return {
        'labels': labels.tolist(),
        'silhouette_score': silhouette_avg,
        'n_clusters': n_clusters
    }


def perform_dbscan_optimized(features_normalized: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Dict:
    """
    Optimized DBSCAN clustering
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_normalized)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    silhouette_avg = None
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            silhouette_avg = silhouette_score(features_normalized[mask], labels[mask])
    
    return {
        'labels': labels.tolist(),
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette_avg,
        'eps': eps,
        'min_samples': min_samples
    }


def extract_cluster_profiles_optimized(
    feature_matrix: pd.DataFrame, 
    cluster_labels: List[int], 
    cluster_name_prefix: str = 'Cluster'
) -> Dict[int, Dict]:
    """
    Extract cluster profiles efficiently
    """
    feature_matrix = feature_matrix.copy()
    feature_matrix['cluster'] = cluster_labels
    
    cluster_profiles = {}
    
    for cluster_id in sorted(feature_matrix['cluster'].unique()):
        cluster_data = feature_matrix[feature_matrix['cluster'] == cluster_id]
        
        # Get numeric columns only
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        
        profile = {
            'cluster_id': int(cluster_id),
            'cluster_name': f"{cluster_name_prefix}_{cluster_id}",
            'n_products': len(cluster_data),
            'percentage': float((len(cluster_data) / len(feature_matrix)) * 100),
            'products': cluster_data['product_name'].tolist(),
            'product_ids': cluster_data['product_id'].tolist(),
            'feature_means': cluster_data[numeric_cols].mean().to_dict(),
            'feature_medians': cluster_data[numeric_cols].median().to_dict(),
            'feature_std': cluster_data[numeric_cols].std().to_dict()
        }
        
        profile['characteristics'] = _determine_cluster_characteristics(cluster_data)
        cluster_profiles[cluster_id] = profile
    
    return cluster_profiles


def _determine_cluster_characteristics(cluster_data: pd.DataFrame) -> List[str]:
    """
    Determine cluster characteristics
    """
    characteristics = []
    
    if 'stat_mean' in cluster_data.columns:
        mean_val = cluster_data['stat_mean'].mean()
        q75 = cluster_data['stat_mean'].quantile(0.75)
        q25 = cluster_data['stat_mean'].quantile(0.25)
        
        if mean_val > q75:
            characteristics.append('High sales volume')
        elif mean_val < q25:
            characteristics.append('Low sales volume')
    
    if 'trend_direction' in cluster_data.columns:
        mode_direction = cluster_data['trend_direction'].mode()
        if len(mode_direction) > 0:
            if mode_direction.iloc[0] == 'increasing':
                characteristics.append('Increasing trend')
            elif mode_direction.iloc[0] == 'decreasing':
                characteristics.append('Decreasing trend')
    
    if 'stat_coefficient_variation' in cluster_data.columns:
        cv_mean = cluster_data['stat_coefficient_variation'].mean()
        if cv_mean > 1:
            characteristics.append('High variability')
        elif cv_mean < 0.3:
            characteristics.append('Low variability')
    
    if 'main_period' in cluster_data.columns and cluster_data['main_period'].mean() > 0:
        characteristics.append('Seasonal pattern')
    
    if 'anomaly_ratio' in cluster_data.columns and cluster_data['anomaly_ratio'].mean() > 0.05:
        characteristics.append('Frequent anomalies')
    
    if 'is_stationary' in cluster_data.columns and cluster_data['is_stationary'].mean() > 0.7:
        characteristics.append('Stationary process')
    
    return characteristics


# ============================================
# 8. MAIN PIPELINE FUNCTIONS
# ============================================

def run_cluster_analysis_pipeline(
    df: pd.DataFrame,
    frequency: str = 'D',
    dbscan_eps: float = 1.0,
    dbscan_min_samples: int = 3,
    progress_callback=None
) -> Dict:
    """
    Complete clustering analysis pipeline
    """
    # 1. Aggregate data
    aggregated_data = aggregate_sales_by_time_optimized(df, frequency)
    
    # 2. Extract product series
    product_series_list = extract_product_series_optimized(aggregated_data)
    
    # 3. Analyze all products
    analyses_results = analyze_all_products_optimized(product_series_list, progress_callback=progress_callback)
    
    # 4. Create feature matrix
    feature_matrix = create_feature_matrix_optimized(analyses_results)
    
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
        'kmeans_results': kmeans_results,
        'hierarchical_results': hierarchical_results,
        'dbscan_results': dbscan_results,
        'cluster_summary': cluster_summary,
    }


def run_time_series_extraction_pipeline(df: pd.DataFrame, frequency: str = 'D') -> List[Dict]:
    """
    Extract time series summaries for all products
    """
    aggregated_data = aggregate_sales_by_time_optimized(df, frequency)
    product_series_list = extract_product_series_optimized(aggregated_data)
    
    results = []
    for product_series in product_series_list:
        summary = get_product_summary_optimized(product_series)
        if summary:
            results.append(summary)
    
    return results
    """
    Determine the main characteristics of a cluster based on its features
    """
    characteristics = []
    
    # Check for high volume products
    if cluster_data['stat_mean'].mean() > cluster_data['stat_mean'].quantile(0.75):
        characteristics.append('High sales volume')
    elif cluster_data['stat_mean'].mean() < cluster_data['stat_mean'].quantile(0.25):
        characteristics.append('Low sales volume')
    
    # Check for trend patterns
    if cluster_data['trend_trend_direction'].mode().iloc[0] == 'increasing':
        characteristics.append('Increasing trend')
    elif cluster_data['trend_trend_direction'].mode().iloc[0] == 'decreasing':
        characteristics.append('Decreasing trend')
    
    # Check for variability
    if cluster_data['stat_coefficient_variation'].mean() > 1:
        characteristics.append('High variability')
    elif cluster_data['stat_coefficient_variation'].mean() < 0.3:
        characteristics.append('Low variability')
    
    # Check for seasonality
    if cluster_data['main_period'].mean() > 0:
        characteristics.append('Seasonal pattern')
    
    # Check for anomalies
    if cluster_data['anomaly_ratio'].mean() > 0.05:
        characteristics.append('Frequent anomalies')
    
    # Check for stability
    if cluster_data['is_stationary'].mean() > 0.7:
        characteristics.append('Stationary process')
    
    return characteristics