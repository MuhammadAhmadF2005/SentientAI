import pandas as pd
from typing import List

class DataCleaner:
    """
    Handles CSV data cleaning tasks following OOP principles.
    """
    def __init__(self, critical_columns: List[str] = None):
        """
        Initializes the DataCleaner with the columns that are critical for analysis.
        """
        self.critical_columns = critical_columns or ['revenue', 'date']

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input sales DataFrame by running sub-tasks:
        1. Standardize column names (strip whitespace and lowercase)
        2. Drop duplicate rows
        3. Check and drop rows with nulls in critical columns
        """
        df = df.copy()
        df = self._standardize_columns(df)
        df = self._drop_duplicates(df)
        df = self._drop_nulls(df)
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strips whitespace and converts column names to lowercase.
        """
        df.columns = [str(col).strip().lower() for col in df.columns]
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops duplicate rows.
        """
        return df.drop_duplicates()

    def _drop_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures critical columns are present and drops nulls in those columns.
        """
        for col in self.critical_columns:
            if col not in df.columns:
                raise ValueError(f"Critical column '{col}' is missing from the data. Columns found: {list(df.columns)}")
        return df.dropna(subset=self.critical_columns)
