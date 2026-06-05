# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from io import StringIO
import logging

from models import (
    ClusterAnalysisResponse, 
    TimeSeriesExtractResponse,
    ProductTimeSeriesAnalysis,
    ProductTimeSeriesSummary,
    KMeansResults,
    HierarchicalResults,
    DBSCANResults,
    ClusterProfile
)
from ts_service import (
    run_cluster_analysis_pipeline,
    run_time_series_extraction_pipeline,
    convert_numpy_nativo
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Time-Series Analysis for Items",
    description="Optimized API for time-series analysis of product sales",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_file_content(df: pd.DataFrame) -> None:
    """Validate file content and required fields"""
    required_fields = ['id_product', 'name', 'category', 'amount', 'time_sale']
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required fields: {missing_fields}. Required: {required_fields}"
        )
    
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="File contains no data")


async def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """Read and parse uploaded file"""
    content = await file.read()
    
    if file.filename.endswith('.json'):
        try:
            data = json.loads(content)
            if isinstance(data, dict) and 'products' in data:
                data = data['products']
            return pd.DataFrame(data)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    
    elif file.filename.endswith('.csv'):
        try:
            return pd.read_csv(StringIO(content.decode('utf-8')))
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use CSV or JSON")


@app.get("/")
async def root():
    return {
        "service": "Time-Series Analysis for Items",
        "version": "2.0.0",
        "description": "Optimized API for time-series analysis of product sales",
        "features": [
            "Product time-series analysis",
            "Trend and seasonality detection",
            "Anomaly detection",
            "Product clustering based on sales patterns"
        ]
    }


@app.post("/run_ts_clusters_analysis", response_model=ClusterAnalysisResponse)
async def run_ts_clusters_analysis(
    file: UploadFile = File(..., description="CSV or JSON with product sales data"),
    frequency: str = "D",
    dbscan_eps: float = 1.0,
    dbscan_min_samples: int = 3
):
    """
    Computes time-series analysis and clustering for product sales data
    
    **File format requirements:**
    - id_product (int): Product identifier
    - name (str): Product name
    - category (str): Product category
    - amount (float): Sale amount
    - time_sale (datetime): Sale timestamp
    
    **Parameters:**
    - frequency: Aggregation frequency (D=day, W=week, M=month)
    - dbscan_eps: DBSCAN epsilon parameter
    - dbscan_min_samples: DBSCAN minimum samples parameter
    """
    try:
        # Read and validate file
        df = await read_uploaded_file(file)
        validate_file_content(df)
        
        # Convert time_sale to datetime
        df['time_sale'] = pd.to_datetime(df['time_sale'])
        
        # Run analysis pipeline
        logger.info(f"Starting cluster analysis with {len(df)} records")
        
        def progress_callback(current, total):
            if current % 100 == 0:
                logger.info(f"Progress: {current}/{total} products analyzed")
        
        results = run_cluster_analysis_pipeline(
            df, 
            frequency=frequency,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            progress_callback=progress_callback
        )
        
        # Convert to JSON serializable
        clean_results = convert_numpy_nativo(results)
        
        logger.info(f"Analysis complete. Found {len(clean_results['cluster_summary'])} clusters")
        
        return JSONResponse(content=jsonable_encoder(clean_results))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/extract_products_time_series", response_model=TimeSeriesExtractResponse)
async def extract_products_time_series(
    file: UploadFile = File(..., description="CSV or JSON with product sales data"),
    frequency: str = "D"
):
    """
    Extracts time-series summaries for each product
    
    Returns for each product:
    - Trend direction (increasing/decreasing/stable)
    - Stationarity status
    - Significant periodic patterns
    - Sales date range
    """
    try:
        # Read and validate file
        df = await read_uploaded_file(file)
        validate_file_content(df)
        
        # Convert time_sale to datetime
        df['time_sale'] = pd.to_datetime(df['time_sale'])
        
        # Run extraction pipeline
        logger.info(f"Starting time-series extraction with {len(df)} records")
        results = run_time_series_extraction_pipeline(df, frequency=frequency)
        
        # Convert to JSON serializable
        clean_results = convert_numpy_nativo({'time_series_results': results})
        
        logger.info(f"Extraction complete. Processed {len(results)} products")
        
        return JSONResponse(content=jsonable_encoder(clean_results))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "endpoints": {
            "/run_ts_clusters_analysis": {
                "method": "POST",
                "description": "Full cluster analysis",
                "supported_formats": ["CSV", "JSON"]
            },
            "/extract_products_time_series": {
                "method": "POST", 
                "description": "Time series extraction only",
                "supported_formats": ["CSV", "JSON"]
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")