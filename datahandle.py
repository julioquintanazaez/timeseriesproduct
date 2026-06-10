
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import HTTPException

# ============================================
# 7. UTILITY FUNCTIONS
# ============================================

def validate_file_content(
        required_fields,
        df: pd.DataFrame) -> None:
    """Validate file content and required fields"""
    #required_fields = ['id_product', 'name', 'category', 'amount', 'time_sale']
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required fields: {missing_fields}. Required: {required_fields}"
        )
    
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="File contains no data")


def convert_numpy_nativo(obj: Any) -> Any:
    """Convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_nativo(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_nativo(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_nativo(item) for item in obj)
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


def numpy_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON serializable"""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    return obj
