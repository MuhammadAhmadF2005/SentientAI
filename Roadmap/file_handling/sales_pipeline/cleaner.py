import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input sales DataFrame:
    - Strips whitespace from column names
    - Converts column names to lowercase
    - Drops duplicate rows
    - Drops nulls in critical columns ('revenue', 'date')
    - Returns the cleaned DataFrame
    """
    # Create a copy to prevent modifying the original DataFrame in place
    df = df.copy()
    
    # Clean column names: strip whitespace and lowercase
    df.columns = [str(col).strip().lower() for col in df.columns]
    
    # Drop completely duplicate rows
    df = df.drop_duplicates()
    
    # Ensure critical columns exist
    critical_cols = ['revenue', 'date']
    for col in critical_cols:
        if col not in df.columns:
            raise ValueError(f"Critical column '{col}' is missing from the data. Columns found: {list(df.columns)}")
    
    # Drop rows with nulls in critical columns
    df = df.dropna(subset=critical_cols)
    
    return df
