import pandas as pd
from typing import Dict, Any

def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarizes the sales DataFrame:
    - Total revenue by region/country
    - Top 5 products by revenue
    - Total rows and total files processed
    - Date range of the data
    """
    if df.empty:
        return {
            "total_rows": 0,
            "total_files": 0,
            "revenue_by_country": {},
            "top_products": {},
            "date_range": {"min": None, "max": None}
        }
    
    # Copy DataFrame to avoid side effects
    df = df.copy()
    
    # Ensure revenue is numeric for grouping and summing
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    
    # 1. Total rows and files processed
    total_rows = len(df)
    total_files = df['source_file'].nunique() if 'source_file' in df.columns else 0
    
    # 2. Revenue by country / region
    # Look for common column names: 'country' or 'region'
    country_col = 'country'
    for col in ['country', 'region']:
        if col in df.columns:
            country_col = col
            break
            
    if country_col in df.columns:
        # Group, sum, and convert numpy float to python float
        raw_revenue = df.groupby(country_col)['revenue'].sum().to_dict()
        revenue_by_country = {str(k): float(v) for k, v in raw_revenue.items()}
    else:
        revenue_by_country = {}
        
    # 3. Top 5 products by revenue
    # Look for common column names: 'product' or 'item'
    product_col = 'product'
    for col in ['product', 'item']:
        if col in df.columns:
            product_col = col
            break
            
    if product_col in df.columns:
        raw_products = df.groupby(product_col)['revenue'].sum().sort_values(ascending=False).head(5).to_dict()
        top_products = {str(k): float(v) for k, v in raw_products.items()}
    else:
        top_products = {}
        
    # 4. Date range of the data
    date_range = {"min": None, "max": None}
    if 'date' in df.columns:
        try:
            # Convert to datetime to handle sorting properly
            dates = pd.to_datetime(df['date'], errors='coerce')
            min_date = dates.min()
            max_date = dates.max()
            
            if pd.notnull(min_date):
                date_range["min"] = min_date.strftime('%Y-%m-%d')
            if pd.notnull(max_date):
                date_range["max"] = max_date.strftime('%Y-%m-%d')
        except Exception:
            # Fallback if datetime conversion fails
            min_val = df['date'].min()
            max_val = df['date'].max()
            date_range["min"] = str(min_val) if pd.notnull(min_val) else None
            date_range["max"] = str(max_val) if pd.notnull(max_val) else None
            
    return {
        "total_rows": total_rows,
        "total_files": total_files,
        "revenue_by_country": revenue_by_country,
        "top_products": top_products,
        "date_range": date_range
    }
