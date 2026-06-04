
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series analysis libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy import stats
from scipy.signal import find_peaks

# Clustering and feature extraction
from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Wavelet transforms
import pywt

# ============================================
# 1. FORMATTING DATA 
# ============================================

def convert_numpy_nativo(obj):
    """Convierte tipos numpy (int64, float64, etc.) a tipos nativos de Python."""
    if isinstance(obj, dict):
        return {k: convert_numpy_nativo(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_nativo(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_nativo(item) for item in obj)
    elif isinstance(obj, set):
        return {convert_numpy_nativo(item) for item in obj}
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.datetime64):
        return obj.astype(datetime)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_nativo(obj.tolist())
    else:
        return obj
    
# ============================================
# 2. DATA PREPROCESSING FUNCTIONS
# ============================================

def aggregate_sales_by_time(sales_df, frequency='D'):
    """
    Aggregate sales data to regular time intervals
    """
    # Ensure datetime
    sales_df = sales_df.copy()
    sales_df['time_sale'] = pd.to_datetime(sales_df['time_sale'])
    
    # Group by product and time
    grouped = sales_df.groupby(['id_product', pd.Grouper(key='time_sale', freq=frequency)])
    
    # Create aggregated features
    aggregated = grouped.agg({
        'amount': ['count', 'mean', 'std', 'min', 'max'],
        'name': 'first'
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = ['id_product', 'time_sale', 'sales_count', 'avg_amount', 
                          'amount_std', 'min_amount', 'max_amount', 'product_name']
    
    # Fill missing dates with zeros
    all_dates = pd.date_range(aggregated['time_sale'].min(), 
                              aggregated['time_sale'].max(), 
                              freq=frequency)
    
    # Create complete time series for each product
    complete_series = []
    for product_id in aggregated['id_product'].unique():
        product_data = aggregated[aggregated['id_product'] == product_id].copy()
        product_data = product_data.set_index('time_sale').reindex(all_dates).reset_index()
        product_data.rename(columns={'index': 'time_sale'}, inplace=True)
        product_data['id_product'] = product_id

        product_data['product_name'] = product_data['product_name'].ffill()  # Updated method
        product_data['sales_count'] = product_data['sales_count'].fillna(0) # Fill sales count with zeros

        # Interpolate price fields
        for col in ['avg_amount', 'amount_std', 'min_amount', 'max_amount']:
            if col in product_data.columns:
                product_data[col] = product_data[col].interpolate(method='linear').bfill()
        
        complete_series.append(product_data)
    
    return pd.concat(complete_series, ignore_index=True)

def extract_time_series_for_product(product_data, value_column='sales_count'):
    """
    Extract time series values for a specific product
    """
    series = product_data.sort_values('time_sale')[value_column].values
    dates = product_data.sort_values('time_sale')['time_sale'].values
    return series, dates


# ============================================
# 3. INDIVIDUAL TIME SERIES ANALYSIS FUNCTIONS
# ============================================

def test_stationarity(series, significance_level=0.05):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    """
    result = adfuller(series, autolag='AIC')
    
    is_stationary = result[1] < significance_level
    
    return {
        'adf_statistic': float(result[0]),
        'p_value': float(result[1]),
        'critical_values': result[4],
        'is_stationary': is_stationary,
        'used_lag': result[2]
    }

def decompose_time_series(series, period=None):
    """
    Decompose time series into trend, seasonal, and residual components
    """
    # Auto-detect period if not provided
    if period is None:
        # Try to find period using autocorrelation
        if len(series) >= 14:
            period = 7  # Weekly seasonality for daily data
        else:
            period = max(2, len(series) // 4)
    
    try:
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
        return {
            'trend': decomposition.trend,
            'period': period
        }
    except:
        return None

def detect_trend(series):
    """
    Detect and quantify trend in time series
    """
    x = np.arange(len(series))
    
    # Linear trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
    
    # Mann-Kendall trend test (non-parametric)
    from scipy.stats import mstats
    # Simplified Kendall tau
    tau, p_value_kendall = stats.kendalltau(x, series)
    
    trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value**2),
        'p_value_linear': float(p_value),
        'kendall_tau': float(tau),
        'p_value_kendall': float(p_value_kendall),
        'trend_direction': trend_direction,
        'magnitude_per_period': float(abs(slope))
    }

def detect_periodicity(series, max_lags=50):
    """
    Detect periodic patterns using autocorrelation
    """
    # Calculate autocorrelation
    acf_values = acf(series, nlags=min(max_lags, len(series)//2), fft=False)
    
    # Find peaks in ACF (excluding lag 0)
    peaks, properties = find_peaks(acf_values[1:], height=0.2)
    peaks = peaks + 1  # Adjust for excluding lag 0
    
    significant_periods = []
    for peak in peaks:
        if acf_values[peak] > 0.3:  # Significant correlation threshold
            significant_periods.append(peak)
    
    # Calculate partial autocorrelation
    pacf_values = pacf(series, nlags=min(max_lags, len(series)//2))
    
    return {
        'significant_periods': significant_periods,
        'main_period': significant_periods[0] if significant_periods else None
    }

def calculate_statistical_features(series):
    """
    Calculate basic statistical features of the time series
    """
    return {
        'mean': float(np.mean(series)),
        'median': float(np.median(series)),
        'std': float(np.std(series)),
        'variance': float(np.var(series)),
        'skewness': float(stats.skew(series)),
        'kurtosis': float(stats.kurtosis(series)),
        'min': float(np.min(series)),
        'max': float(np.max(series)),
        'range': float(np.max(series) - np.min(series)),
        'q25': float(np.percentile(series, 25)),
        'q50': float(np.percentile(series, 50)),
        'q75': float(np.percentile(series, 75)),
        'iqr': float(np.percentile(series, 75) - np.percentile(series, 25)),
        'coefficient_variation': float(np.std(series) / (np.mean(series) + 1e-8)),
        'zero_crossings': float(sum(np.diff(np.signbit(series)))),
        'abs_energy': float(np.sum(series**2))
    }

def detect_anomalies(series, threshold=3):
    """
    Detect anomalies using Z-score method
    """
    z_scores = np.abs(stats.zscore(series))
    anomalies = z_scores > threshold
    
    return {
        'anomaly_indices': np.where(anomalies)[0].tolist(),
        'anomaly_count': float(np.sum(anomalies)),
        'anomaly_ratio': float(np.sum(anomalies) / len(series)),
    }

def analyze_product_time_series(product_id, aggregated_data):
    """
    Perform comprehensive time series analysis for a single product
    """
    product_data = aggregated_data[aggregated_data['id_product'] == product_id]
    series, dates = extract_time_series_for_product(product_data, 'sales_count')
    
    if len(series) < 7:  # Need at least 7 points for meaningful analysis
        return None
    
    # Perform various analyses
    stationarity = convert_numpy_nativo(test_stationarity(series))
    #decomposition = decompose_time_series(series)
    trend = detect_trend(series)
    periodicity = convert_numpy_nativo(detect_periodicity(series))
    stats_features = calculate_statistical_features(series)
    anomalies = detect_anomalies(series)
    date_range = convert_numpy_nativo({
            'start': dates[0],
            'end': dates[-1]
        })
    
    # Summary statistics
    analysis = {
        'product_id': str(product_id),
        'product_name': product_data['product_name'].iloc[0],
        'series_length': len(series),
        'date_range': date_range,
        'stationarity': stationarity,
        'trend': trend,
        'periodicity': periodicity,
        'statistical_features': stats_features,
        'anomalies': anomalies,
        'raw_series': series,
    }
    
    return analysis


def get_products_time_series(product_id, aggregated_data):
    """
    Perform comprehensive time series analysis for a single product
    """
    product_data = aggregated_data[aggregated_data['id_product'] == product_id]
    series, dates = extract_time_series_for_product(product_data, 'sales_count')
    
    if len(series) < 7:  # Need at least 7 points for meaningful analysis
        return None
    
    # Perform various analyses
    stationarity = convert_numpy_nativo(test_stationarity(series))
    trend = detect_trend(series)
    periodicity = detect_periodicity(series)
    date_range = convert_numpy_nativo({
            'start': dates[0],
            'end': dates[-1]
        })
    
    # Summary statistics
    analysis = {
        'product_id': str(product_id),
        'product_name': product_data['product_name'].iloc[0],
        'series_length': len(series),
        'date_range': date_range,
        'stationarity': stationarity.get("is_stationary"),
        'critical_values': stationarity.get("critical_values"),
        'trend': trend.get("trend_direction"),
        'significant_periods': periodicity.get("significant_periods")
    }
    
    return analysis


# ============================================
# 4. FEATURE EXTRACTION FUNCTIONS
# ============================================

def extract_tsfresh_features(aggregated_data, column_id='id_product', 
                            column_sort='time_sale', column_value='sales_count'):
    """
    Extract comprehensive features using tsfresh
    """
    # Prepare data for tsfresh
    df_features = aggregated_data[[column_id, column_sort, column_value]].copy()
    df_features.rename(columns={column_value: 'value'}, inplace=True)
    
    # Extract features
    extraction_settings = EfficientFCParameters()
    features = extract_features(df_features, 
                                column_id=column_id, 
                                column_sort=column_sort,
                                default_fc_parameters=extraction_settings,
                                disable_progressbar=True)
    
    # Handle NaN values
    features = features.fillna(0)
    
    # Add product names
    product_names = aggregated_data.groupby('id_product')['product_name'].first()
    features['product_name'] = product_names
    
    return features

def extract_wavelet_features(series, wavelet='db4', level=3):
    """
    Extract features using Discrete Wavelet Transform (DWT)
    """
    # Apply wavelet decomposition
    coeffs = pywt.wavedec(series, wavelet, level=min(level, pywt.dwt_max_level(len(series), wavelet)))
    
    # Extract features from coefficients
    features = {}
    
    # Energy at each decomposition level
    for i, coeff in enumerate(coeffs):
        level_name = 'approx' if i == 0 else f'detail_{i}'
        energy = np.sum(coeff**2)
        features[f'wavelet_{level_name}_energy'] = energy
        features[f'wavelet_{level_name}_mean'] = np.mean(np.abs(coeff))
        features[f'wavelet_{level_name}_std'] = np.std(coeff)
    
    # Statistical features of wavelet coefficients
    all_coeffs = np.concatenate(coeffs)
    features['wavelet_total_energy'] = np.sum(all_coeffs**2)
    features['wavelet_entropy'] = -np.sum((coeffs[0]**2) / (features['wavelet_total_energy'] + 1e-8) * 
                                          np.log((coeffs[0]**2) / (features['wavelet_total_energy'] + 1e-8) + 1e-8))
    
    return features

def create_feature_matrix(analyses_results, aggregated_data):
    """
    Create comprehensive feature matrix from all products
    Maneja tanto diccionarios individuales como listas de diccionarios
    """
    feature_matrix = []
    
    # Determinar la estructura de datos
    if isinstance(analyses_results, dict):
        # Caso 1: Es un diccionario con un solo producto o múltiples productos
        # Verificar si es un producto individual o un diccionario de productos
        if 'product_id' in analyses_results:
            # Es un solo producto
            products_to_process = [analyses_results]
        else:
            # Es un diccionario de productos {id: analysis}
            products_to_process = list(analyses_results.values())
    
    elif isinstance(analyses_results, list):
        # Caso 2: Es una lista de productos
        products_to_process = analyses_results
    
    else:
        raise TypeError(f"Tipo no soportado: {type(analyses_results)}")
    
    print(f"Procesando {len(products_to_process)} productos...")
    
    for analysis in products_to_process:
        if analysis is None:
            continue
        
        # Convertir numpy types a tipos nativos
        def to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val
        
        features = {}
        features['product_id'] = str(to_native(analysis.get('product_id', '')))
        features['product_name'] = str(analysis.get('product_name', ''))
        
        # Add statistical features
        if 'statistical_features' in analysis:
            stats = analysis['statistical_features']
            for key, value in stats.items():
                features[f'stat_{key}'] = to_native(value)
        
        # Add trend features (excluyendo algunos)
        if 'trend' in analysis:
            trend = analysis['trend']
            exclude_keys = ['r_squared', 'p_value_linear', 'p_value_kendall']
            for key, value in trend.items():
                if key not in exclude_keys:
                    features[f'trend_{key}'] = to_native(value)
        
        # Add stationarity features
        if 'stationarity' in analysis:
            stationarity = analysis['stationarity']
            features['stationarity_pvalue'] = to_native(stationarity.get('p_value', 0))
            features['is_stationary'] = stationarity.get('is_stationary', False)
            features['adf_statistic'] = to_native(stationarity.get('adf_statistic', 0))
        
        # Add periodicity features
        if 'periodicity' in analysis:
            periodicity = analysis['periodicity']
            main_period = periodicity.get('main_period')
            features['main_period'] = to_native(main_period) if main_period is not None else 0
            features['n_significant_periods'] = len(periodicity.get('significant_periods', []))
        else:
            features['main_period'] = 0
            features['n_significant_periods'] = 0
        
        # Add anomaly features
        if 'anomalies' in analysis:
            anomalies = analysis['anomalies']
            features['anomaly_count'] = to_native(anomalies.get('anomaly_count', 0))
            features['anomaly_ratio'] = to_native(anomalies.get('anomaly_ratio', 0.0))
        else:
            features['anomaly_count'] = 0
            features['anomaly_ratio'] = 0.0
        
        # Add series metadata
        features['series_length'] = to_native(analysis.get('series_length', 0))
        
        # Add wavelet features (si la función existe)
        if 'raw_series' in analysis and analysis['raw_series'] is not None:
            try:
                wavelet_features = extract_wavelet_features(analysis['raw_series'])
                features.update(wavelet_features)
            except NameError:
                print("Warning: extract_wavelet_features no está definida")
            except Exception as e:
                print(f"Error en wavelet features: {e}")
        
        feature_matrix.append(features)
    
    return pd.DataFrame(feature_matrix)

# ============================================
# 4. FEATURE EXTRACTION FUNCTIONS
# ============================================

def extract_tsfresh_features(aggregated_data, column_id='id_product', 
                            column_sort='time_sale', column_value='sales_count'):
    """
    Extract comprehensive features using tsfresh
    """
    # Prepare data for tsfresh
    df_features = aggregated_data[[column_id, column_sort, column_value]].copy()
    df_features.rename(columns={column_value: 'value'}, inplace=True)
    
    # Extract features
    extraction_settings = EfficientFCParameters()
    features = extract_features(df_features, 
                                column_id=column_id, 
                                column_sort=column_sort,
                                default_fc_parameters=extraction_settings,
                                disable_progressbar=True)
    
    # Handle NaN values
    features = features.fillna(0)
    
    # Add product names
    product_names = aggregated_data.groupby('id_product')['product_name'].first()
    features['product_name'] = product_names
    
    return features

def extract_wavelet_features(series, wavelet='db4', level=3):
    """
    Extract features using Discrete Wavelet Transform (DWT)
    """
    # Apply wavelet decomposition
    coeffs = pywt.wavedec(series, wavelet, level=min(level, pywt.dwt_max_level(len(series), wavelet)))
    
    # Extract features from coefficients
    features = {}
    
    # Energy at each decomposition level
    for i, coeff in enumerate(coeffs):
        level_name = 'approx' if i == 0 else f'detail_{i}'
        energy = np.sum(coeff**2)
        features[f'wavelet_{level_name}_energy'] = energy
        features[f'wavelet_{level_name}_mean'] = np.mean(np.abs(coeff))
        features[f'wavelet_{level_name}_std'] = np.std(coeff)
    
    # Statistical features of wavelet coefficients
    all_coeffs = np.concatenate(coeffs)
    features['wavelet_total_energy'] = np.sum(all_coeffs**2)
    features['wavelet_entropy'] = -np.sum((coeffs[0]**2) / (features['wavelet_total_energy'] + 1e-8) * 
                                          np.log((coeffs[0]**2) / (features['wavelet_total_energy'] + 1e-8) + 1e-8))
    
    return features

def create_feature_matrix(analyses_results, aggregated_data):
    """
    Create comprehensive feature matrix from all products
    Maneja tanto diccionarios individuales como listas de diccionarios
    """
    feature_matrix = []
    
    # Determinar la estructura de datos
    if isinstance(analyses_results, dict):
        # Caso 1: Es un diccionario con un solo producto o múltiples productos
        # Verificar si es un producto individual o un diccionario de productos
        if 'product_id' in analyses_results:
            # Es un solo producto
            products_to_process = [analyses_results]
        else:
            # Es un diccionario de productos {id: analysis}
            products_to_process = list(analyses_results.values())
    
    elif isinstance(analyses_results, list):
        # Caso 2: Es una lista de productos
        products_to_process = analyses_results
    
    else:
        raise TypeError(f"Tipo no soportado: {type(analyses_results)}")
    
    print(f"Procesando {len(products_to_process)} productos...")
    
    for analysis in products_to_process:
        if analysis is None:
            continue
        
        # Convertir numpy types a tipos nativos
        def to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val
        
        features = {}
        features['product_id'] = str(to_native(analysis.get('product_id', '')))
        features['product_name'] = str(analysis.get('product_name', ''))
        
        # Add statistical features
        if 'statistical_features' in analysis:
            stats = analysis['statistical_features']
            for key, value in stats.items():
                features[f'stat_{key}'] = to_native(value)
        
        # Add trend features (excluyendo algunos)
        if 'trend' in analysis:
            trend = analysis['trend']
            exclude_keys = ['r_squared', 'p_value_linear', 'p_value_kendall']
            for key, value in trend.items():
                if key not in exclude_keys:
                    features[f'trend_{key}'] = to_native(value)
        
        # Add stationarity features
        if 'stationarity' in analysis:
            stationarity = analysis['stationarity']
            features['stationarity_pvalue'] = to_native(stationarity.get('p_value', 0))
            features['is_stationary'] = stationarity.get('is_stationary', False)
            features['adf_statistic'] = to_native(stationarity.get('adf_statistic', 0))
        
        # Add periodicity features
        if 'periodicity' in analysis:
            periodicity = analysis['periodicity']
            main_period = periodicity.get('main_period')
            features['main_period'] = to_native(main_period) if main_period is not None else 0
            features['n_significant_periods'] = len(periodicity.get('significant_periods', []))
        else:
            features['main_period'] = 0
            features['n_significant_periods'] = 0
        
        # Add anomaly features
        if 'anomalies' in analysis:
            anomalies = analysis['anomalies']
            features['anomaly_count'] = to_native(anomalies.get('anomaly_count', 0))
            features['anomaly_ratio'] = to_native(anomalies.get('anomaly_ratio', 0.0))
        else:
            features['anomaly_count'] = 0
            features['anomaly_ratio'] = 0.0
        
        # Add series metadata
        features['series_length'] = to_native(analysis.get('series_length', 0))
        
        # Add wavelet features (si la función existe)
        if 'raw_series' in analysis and analysis['raw_series'] is not None:
            try:
                wavelet_features = extract_wavelet_features(analysis['raw_series'])
                features.update(wavelet_features)
            except NameError:
                print("Warning: extract_wavelet_features no está definida")
            except Exception as e:
                print(f"Error en wavelet features: {e}")
        
        feature_matrix.append(features)
    
    return pd.DataFrame(feature_matrix)


# ============================================
# 5. CLUSTERING FUNCTIONS
# ============================================

def normalize_features(feature_matrix, exclude_columns=['product_id', 'product_name', 'is_stationary', 'trend_trend_direction']):
    """
    Normalize features for clustering
    """
    feature_cols = [col for col in feature_matrix.columns if col not in exclude_columns]
    
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix[feature_cols])
    
    return normalized_features, scaler, feature_cols

def perform_kmeans_clustering(features_normalized, n_clusters_range=range(2, 11)):
    """
    Perform K-means clustering with optimal cluster selection
    """
    best_k = 2
    best_silhouette = -1
    
    kmeans_models = {}
    silhouette_scores = {}
    inertia_values = {}
    
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_normalized)
        
        silhouette = silhouette_score(features_normalized, labels)
        silhouette_scores[k] = silhouette
        
        inertia_values[k] = kmeans.inertia_
        kmeans_models[k] = kmeans
        
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
    
    # Final model with best k
    best_kmeans = kmeans_models[best_k]
    final_labels = best_kmeans.predict(features_normalized)
    
    return {
        'model': best_kmeans,
        'labels': final_labels,
        'optimal_k': best_k,
        'silhouette_scores': silhouette_scores,
        'inertia_values': inertia_values,
        'all_models': kmeans_models
    }

def perform_hierarchical_clustering(features_normalized, n_clusters=5):
    """
    Perform hierarchical clustering
    """
    # Compute linkage matrix
    linkage_matrix = linkage(features_normalized, method='ward')
    
    # Perform clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(features_normalized)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(features_normalized, labels)
    
    return {
        'labels': labels,
        #'linkage_matrix': linkage_matrix,
        'silhouette_score': silhouette_avg,
        'model': hierarchical,
        'n_clusters': n_clusters
    }

def perform_dbscan_clustering(features_normalized, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_normalized)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Calculate silhouette score if more than 1 cluster
    silhouette_avg = None
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            silhouette_avg = silhouette_score(features_normalized[mask], labels[mask])
    
    return {
        'labels': labels,
        'model': dbscan,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette_avg,
        'eps': eps,
        'min_samples': min_samples
    }

def extract_cluster_profiles(feature_matrix, cluster_labels, cluster_name_prefix='Cluster'):
    """
    Extract and describe profiles of each cluster
    """
    feature_matrix = feature_matrix.copy()
    feature_matrix['cluster'] = cluster_labels
    
    cluster_profiles = {}
    
    for cluster_id in sorted(feature_matrix['cluster'].unique()):
        cluster_data = feature_matrix[feature_matrix['cluster'] == cluster_id]
        
        # Calculate summary statistics for each feature
        profile = {
            'cluster_id': cluster_id,
            'cluster_name': f"{cluster_name_prefix}_{cluster_id}",
            'n_products': len(cluster_data),
            'percentage': (len(cluster_data) / len(feature_matrix)) * 100,
            'products': cluster_data['product_name'].tolist(),
            'product_ids': cluster_data['product_id'].tolist(),
            'feature_means': cluster_data.select_dtypes(include=[np.number]).mean().to_dict(),
            'feature_medians': cluster_data.select_dtypes(include=[np.number]).median().to_dict(),
            'feature_std': cluster_data.select_dtypes(include=[np.number]).std().to_dict()
        }
        
        # Determine cluster characteristics
        profile['characteristics'] = determine_cluster_characteristics(cluster_data)
        
        cluster_profiles[cluster_id] = profile
    
    return cluster_profiles

def determine_cluster_characteristics(cluster_data):
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