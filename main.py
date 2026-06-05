# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from io import StringIO
import logging

from fastapi.responses import StreamingResponse

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
    convert_numpy_nativo,
    aggregate_sales_by_time_optimized,
    extract_product_series_optimized
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


@app.post("/extract_raw_time_series")
async def extract_raw_time_series(
    file: UploadFile = File(..., description="CSV or JSON with product sales data"),
    frequency: str = "D",
    max_products: Optional[int] = None
):
    """
    Extracts raw time series for each product (sales count over time)
    
    Returns for each product:
    - product_id: Product identifier
    - product_name: Product name
    - time_series: List of sales counts over time
    - dates: Corresponding dates for each sales point
    - series_length: Length of the time series
    
    **Parameters:**
    - frequency: Aggregation frequency (D=day, W=week, M=month)
    - max_products: Maximum number of products to return (useful for large datasets)
    """
    try:
        # Read and validate file
        df = await read_uploaded_file(file)
        validate_file_content(df)
        
        # Convert time_sale to datetime
        df['time_sale'] = pd.to_datetime(df['time_sale'])
        
        # Aggregate data
        aggregated_data = aggregate_sales_by_time_optimized(df, frequency)
        
        if aggregated_data.empty:
            return JSONResponse(content={
                "message": "No data available after aggregation",
                "total_products": 0,
                "time_series_data": []
            })
        
        # Extract raw time series for each product
        product_series_list = extract_product_series_optimized(aggregated_data, 'sales_count')
        
        # Limit number of products if specified
        if max_products and max_products > 0:
            product_series_list = product_series_list[:max_products]
        
        # Prepare the response - FIXED: Convert numpy.datetime64 to string
        time_series_data = []
        for product_series in product_series_list:
            # Convert numpy datetime64 to Python datetime and then to ISO format
            dates_iso = []
            for d in product_series.dates:
                if isinstance(d, np.datetime64):
                    # Convert numpy.datetime64 to pandas Timestamp then to ISO string
                    dates_iso.append(pd.Timestamp(d).isoformat())
                elif isinstance(d, datetime):
                    dates_iso.append(d.isoformat())
                else:
                    dates_iso.append(str(d))
            
            time_series_data.append({
                'product_id': product_series.product_id,
                'product_name': product_series.product_name,
                'series_length': product_series.length,
                'dates': dates_iso,
                'time_series': product_series.series.tolist(),
                'start_date': pd.Timestamp(product_series.start_date).isoformat() if isinstance(product_series.start_date, np.datetime64) else product_series.start_date.isoformat(),
                'end_date': pd.Timestamp(product_series.end_date).isoformat() if isinstance(product_series.end_date, np.datetime64) else product_series.end_date.isoformat(),
                'summary': {
                    'mean': float(np.mean(product_series.series)),
                    'std': float(np.std(product_series.series)),
                    'min': float(np.min(product_series.series)),
                    'max': float(np.max(product_series.series)),
                    'total_sales': float(np.sum(product_series.series))
                }
            })
        
        logger.info(f"Extracted raw time series for {len(time_series_data)} products")
        
        result = {
            "total_products": len(time_series_data),
            "aggregation_frequency": frequency,
            "time_series_data": time_series_data
        }
        
        return JSONResponse(content=jsonable_encoder(convert_numpy_nativo(result)))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting raw time series: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    


@app.post("/extract_raw_time_series_stream")
async def extract_raw_time_series_stream(
    file: UploadFile = File(..., description="CSV or JSON with product sales data"),
    frequency: str = "D",
    max_products: Optional[int] = None
):
    """
    Extracts raw time series with streaming response for better performance
    """
    try:
        # Read and validate file
        df = await read_uploaded_file(file)
        validate_file_content(df)
        
        # Convert time_sale to datetime
        df['time_sale'] = pd.to_datetime(df['time_sale'])
        
        # Aggregate data
        aggregated_data = aggregate_sales_by_time_optimized(df, frequency)
        
        if aggregated_data.empty:
            return JSONResponse(content={
                "message": "No data available after aggregation",
                "total_products": 0
            })
        
        # Extract raw time series
        product_series_list = extract_product_series_optimized(aggregated_data, 'sales_count')
        
        # Limit number of products if specified
        if max_products and max_products > 0:
            product_series_list = product_series_list[:max_products]
        
        async def generate_stream():
            """Generator that yields JSON chunks"""
            # Send header
            yield '{"total_products": ' + str(len(product_series_list)) + ', '
            yield '"aggregation_frequency": "' + frequency + '", '
            yield '"time_series_data": ['
            
            # Send each product one by one
            for i, product_series in enumerate(product_series_list):
                # Convert dates to ISO format efficiently
                dates_iso = [pd.Timestamp(d).isoformat() for d in product_series.dates]
                
                product_data = {
                    'product_id': product_series.product_id,
                    'product_name': product_series.product_name,
                    'series_length': product_series.length,
                    'dates': dates_iso,
                    'time_series': product_series.series.tolist(),
                    'start_date': pd.Timestamp(product_series.start_date).isoformat(),
                    'end_date': pd.Timestamp(product_series.end_date).isoformat(),
                    'summary': {
                        'mean': float(np.mean(product_series.series)),
                        'std': float(np.std(product_series.series)),
                        'min': float(np.min(product_series.series)),
                        'max': float(np.max(product_series.series)),
                        'total_sales': float(np.sum(product_series.series))
                    }
                }
                
                # Convert to JSON and yield
                json_chunk = json.dumps(product_data, default=str)
                
                # Add comma between items (except last)
                if i < len(product_series_list) - 1:
                    json_chunk += ','
                
                yield json_chunk
                
                # Small delay to avoid memory buildup (optional)
                if i % 10 == 0:
                    await asyncio.sleep(0)
            
            # Send footer
            yield ']}'
        
        return StreamingResponse(
            generate_stream(),
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=time_series_stream.json",
                "X-Content-Type-Options": "nosniff"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting raw time series: {str(e)}", exc_info=True)
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