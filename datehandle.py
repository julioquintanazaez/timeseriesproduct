import pandas as pd
from datetime import datetime
import re

def normalizar_fechas(df, columna_fecha, formato_salida="%Y-%m-%d %H:%M:%S"):
    """
    Normaliza las fechas de una columna a un formato específico.
    
    Parámetros:
    - df: DataFrame de pandas
    - columna_fecha: nombre de la columna que contiene las fechas
    - formato_salida: formato deseado (por defecto: "%Y-%m-%d %H:%M:%S")
    
    Retorna:
    - DataFrame con la columna de fechas normalizada como strings
    """
    
    def parsear_fecha_single(fecha):
        """Parsea una fecha individual en múltiples formatos"""
        
        # Si ya es datetime, formatear directamente
        if isinstance(fecha, (pd.Timestamp, datetime)):
            return fecha.strftime(formato_salida)
        
        # Si es NaN o None
        if pd.isna(fecha):
            return None
        
        # Convertir a string
        fecha_str = str(fecha).strip()
        
        # Caso especial: solo un número (día o timestamp)
        if fecha_str.isdigit():
            # Asumir que es un día del mes actual o timestamp Unix
            if len(fecha_str) == 1 or len(fecha_str) == 2:
                # Día sin mes/año (usar fecha actual como referencia)
                ahora = datetime.now()
                try:
                    fecha_parseada = datetime(ahora.year, ahora.month, int(fecha_str))
                    return fecha_parseada.strftime(formato_salida)
                except:
                    pass
        
        # Lista de formatos posibles (ordenados por especificidad)
        formatos = [
            "%d/%m/%Y %H:%M",      # 1/1/2023 11:14
            "%d/%m/%Y %H:%M:%S",   # 1/1/2023 11:14:00
            "%m/%d/%Y %H:%M",      # 1/1/2023 11:14 (mes/día)
            "%Y-%m-%d %H:%M:%S",   # 2023-01-01 11:14:00
            "%Y-%m-%d %H:%M",      # 2023-01-01 11:14
            "%d/%m/%y %H:%M",      # 1/1/23 11:14
            "%Y/%m/%d %H:%M:%S",   # 2023/01/01 11:14:00
            "%Y/%m/%d %H:%M",      # 2023/01/01 11:14
            "%d-%m-%Y %H:%M:%S",   # 1-1-2023 11:14:00
            "%d-%m-%Y %H:%M",      # 1-1-2023 11:14
            "%Y%m%d %H:%M:%S",     # 20230101 11:14:00
        ]
        
        # Intentar cada formato
        for fmt in formatos:
            try:
                fecha_parseada = datetime.strptime(fecha_str, fmt)
                return fecha_parseada.strftime(formato_salida)
            except ValueError:
                continue
        
        # Intentar con pandas.to_datetime (flexible)
        try:
            fecha_parseada = pd.to_datetime(fecha_str)
            return fecha_parseada.strftime(formato_salida)
        except:
            pass
        
        # Si nada funciona, devolver el original o None
        print(f"Advertencia: No se pudo parsear la fecha: {fecha_str}")
        return None
    
    # Aplicar la función a la columna
    df[columna_fecha] = df[columna_fecha].apply(parsear_fecha_single)
    
    return df


def normalize_dates_optimized(df, columna_fecha, formato_salida="%Y-%m-%d %H:%M:%S"):
    """
    Versión optimizada para usar con apply
    """
    # Pre-compilar formatos como constante de clase para evitar recrearlos
    FORMATOS = [
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%y %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%Y%m%d %H:%M:%S",
    ]
    
    def parsear_fecha(fecha):
        if pd.isna(fecha):
            return None
        
        if isinstance(fecha, (pd.Timestamp, datetime)):
            return fecha.strftime(formato_salida)
        
        fecha_str = str(fecha).strip()
        
        # Caso especial: solo números (día)
        if fecha_str.isdigit() and len(fecha_str) <= 2:
            ahora = datetime.now()
            try:
                return datetime(ahora.year, ahora.month, int(fecha_str)).strftime(formato_salida)
            except:
                pass
        
        # Intentar formatos predefinidos
        for fmt in FORMATOS:
            try:
                return datetime.strptime(fecha_str, fmt).strftime(formato_salida)
            except ValueError:
                continue
        
        # Fallback a pandas (más lento pero cubre casos extremos)
        try:
            return pd.to_datetime(fecha_str).strftime(formato_salida)
        except:
            return None
        
    df[columna_fecha] = df[columna_fecha].apply(parsear_fecha)
    return df
    
   
def normalize_dates_vectorized(df, columna_fecha):
    """
    Versión vectorizada - MUCHO más rápida para DataFrames grandes
    """
    # Convertir a datetime de una vez (pandas es muy inteligente)
    df[columna_fecha] = pd.to_datetime(
        df[columna_fecha],
        errors='coerce',  # Los que no se pueden convertir quedan como NaT
        format='mixed',    # Permite formatos mixtos
        dayfirst=True      # Priorizar día/mes (importante para 1/1/2023)
    )
    
    # Formatear a string
    df[columna_fecha] = df[columna_fecha].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return df

def normalize_dates_vectorized1(df, columna_fecha):
    """
    Normaliza fechas pero devuelve datetime, no string
    """
    df[columna_fecha] = pd.to_datetime(
        df[columna_fecha],
        errors='coerce',
        format='mixed',
        dayfirst=True
    )
    # NO formatear a string - mantener como datetime
    return df