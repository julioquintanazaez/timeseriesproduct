from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd
import json
import numpy as np
from typing import List, Dict
from datetime import datetime
from io import StringIO

from ts_service import (
    aggregate_sales_by_time, 
    normalize_features, 
    extract_cluster_profiles, 
    create_feature_matrix,
    perform_dbscan_clustering, 
    perform_hierarchical_clustering,
    perform_kmeans_clustering,
    convert_numpy_nativo,
    analyze_product_time_series,
    get_products_time_series
    )


app = FastAPI(
    title="Time-Series Analysis for items",
    description="This API is for the uses of time-series algorithms",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "service": "Time-Series Analysis for items",
        "version": "1.0.0",
        "description": "This API uses time-series algorithms to analyse product sales history",
    }

@app.post("/run_ts_clusters_analysis")
async def run_ts_clusters_analysis(
    file: UploadFile = File(..., description="CSV or JSON with products: id_product, name, category, amount, time_sale")
):
    """
    Computes the time-series and clusters for the sales products data  
    
    Expected file format:
    - id_product (int)
    - name (str)
    - category (str)
    - amount (float)
    - time_sale (datetime)
    """
    
    # Read file
    try:
        content = await file.read()
        
        if file.filename.endswith('.json'):
            data = json.loads(content)
            if isinstance(data, dict) and 'products' in data:
                data = data['products']
            df = pd.DataFrame(data)  # Convert to DataFrame consistently
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(400, "Unsupported format. Use JSON or CSV")
        
        # Validate required fields
        required_fields = ['id_product', 'name', 'category', 'amount', 'time_sale']
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            raise HTTPException(400, f"Missing required fields: {missing_fields}. Needs: {required_fields}")
        
        # Convert time_sale to datetime
        df['time_sale'] = pd.to_datetime(df['time_sale'])
        
        # Processing pipeline:
        #----------Aggregte data from input--------------- 
        aggregated_data = aggregate_sales_by_time(df)

        #----------Compute analyzis for each product----------
        product_ids = aggregated_data['id_product'].unique()
        analyses_results = []

        for i, product_id in enumerate(product_ids):
            if (i + 1) % 20 == 0:
                print(f" Progress: {i+1}/{len(product_ids)} products analyzed")
            analysis, series = analyze_product_time_series(product_id, aggregated_data)
            if analysis:
                analyses_results.append(analysis)
        
        #----------Create feature matrix----------
        feature_matrix = create_feature_matrix(analyses_results, aggregated_data)
        
        #----------Prepare data for clustering algorithms---------------
        features_normalized, scaler, feature_cols = normalize_features(feature_matrix)
        
        #----------Apply clustering algorithms-------------
        kmeans_results = perform_kmeans_clustering(features_normalized)
        hierarchical_results = perform_hierarchical_clustering(features_normalized, n_clusters=kmeans_results['optimal_k'])
        dbscan_results = perform_dbscan_clustering(features_normalized, eps=1.0, min_samples=3)

        #----------Extract clusters profiles--------------
        cluster_profiles = extract_cluster_profiles(feature_matrix, kmeans_results['labels'])
        
        #-------------Create cluster profiles summary------------------
        cluster_summary = []
        for cluster_id, profile in cluster_profiles.items():
            cluster_summary.append({
                'cluster_id': cluster_id,
                'cluster_name': profile['cluster_name'],
                'n_products': profile['n_products'],
                'percentage': profile['percentage'],
                'characteristics': ', '.join(profile['characteristics'])
            })

        # Remove from analyses result the raw?series feature
        print("antes de eliminar")
        #for product in analyses_results['analyses_results']:
        #    product.pop('raw_series', None) 
        
        corrected_json = convert_numpy_nativo({
            'analyses_results': analyses_results,
            'kmeans_results': kmeans_results,
            'hierarchical_results': hierarchical_results,
            'dbscan_results': dbscan_results,
            'cluster_summary': cluster_summary,
        })

        print("después de eliminar")
        
        for product in corrected_json['analyses_results']:
            product.pop('raw_series', None)

        clean_result = {}
        for key, value in corrected_json.items():
            if key not in ['kmeans_results', 'hierarchical_results', 'dbscan_results']:
                clean_result[key] = value
            else:
                clean_result[key] = {
                    k: v for k, v in value.items() 
                    if k not in ['model', 'all_models']
                }


        return JSONResponse(content=jsonable_encoder(clean_result))

        
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "File is empty")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON format")
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")
    

@app.post("/extract_products_time_series")
async def extract_products_time_series(
    file: UploadFile = File(..., description="CSV or JSON with products: id_product, name, category, amount, time_sale")
):
    """
    Computes the time-series for each products in sales data  
    
    Expected file format:
    - id_product (int)
    - name (str)
    - category (str)
    - amount (float)
    - time_sale (datetime)

    return: A time series for each product, its tendency and its stacionarity state

    """
    # Read file
    try:
        content = await file.read()
        
        if file.filename.endswith('.json'):
            data = json.loads(content)
            if isinstance(data, dict) and 'products' in data:
                data = data['products']
            df = pd.DataFrame(data)  # Convert to DataFrame consistently
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(400, "Unsupported format. Use JSON or CSV")
        
        # Validate required fields
        required_fields = ['id_product', 'name', 'category', 'amount', 'time_sale']
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            raise HTTPException(400, f"Missing required fields: {missing_fields}. Needs: {required_fields}")
        
        # Convert time_sale to datetime
        df['time_sale'] = pd.to_datetime(df['time_sale'])
        
        # Processing pipeline:
        #----------Aggregte data from input--------------- 
        aggregated_data = aggregate_sales_by_time(df)

        #----------Compute analyzis for each product----------
        product_ids = aggregated_data['id_product'].unique()
        time_series_results = []

        for i, product_id in enumerate(product_ids):
            if (i + 1) % 20 == 0:
                print(f" Progress: {i+1}/{len(product_ids)} products analyzed")
            time_series = get_products_time_series(product_id, aggregated_data)
            if time_series:
                time_series_results.append(time_series)
        
        corrected_json = convert_numpy_nativo({
            'time_series_results': time_series_results
        })
       
        return JSONResponse(content=jsonable_encoder(corrected_json))

        
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "File is empty")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON format")
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")

            

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "methods": {
            "method-1": {"name": "run_ts_clusters_analysis",
                         "supported_formats": ["CSV", "JSON"]
                         }
        }        
    }


# ==================== MAIN (for local development) ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)