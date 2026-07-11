import pandas as pd
from typing import Dict, Any, List

class DataAnalyzer:
    """
    Analyzes sales data using OOP principles, computing metrics and aggregations.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the analyzer with a DataFrame and runs pre-processing.
        """
        self.df = df.copy() if not df.empty else df
        self._preprocess()

    def _preprocess(self) -> None:
        """
        Preprocesses values, ensuring revenue column is numeric.
        """
        if not self.df.empty and 'revenue' in self.df.columns:
            self.df['revenue'] = pd.to_numeric(self.df['revenue'], errors='coerce').fillna(0)

    def get_total_rows(self) -> int:
        """
        Returns the total number of rows.
        """
        return len(self.df)

    def get_total_files(self) -> int:
        """
        Returns the count of distinct source files processed.
        """
        if 'source_file' in self.df.columns:
            return self.df['source_file'].nunique()
        return 0

    def get_revenue_by_country(self) -> Dict[str, float]:
        """
        Calculates total revenue grouped by country or region.
        """
        country_col = self._find_column(['country', 'region'], 'country')
        if country_col in self.df.columns:
            raw_revenue = self.df.groupby(country_col)['revenue'].sum().to_dict()
            return {str(k): float(v) for k, v in raw_revenue.items()}
        return {}

    def get_top_products(self, limit: int = 5) -> Dict[str, float]:
        """
        Finds the top products by revenue.
        """
        product_col = self._find_column(['product', 'item'], 'product')
        if product_col in self.df.columns:
            raw_products = self.df.groupby(product_col)['revenue'].sum().sort_values(ascending=False).head(limit).to_dict()
            return {str(k): float(v) for k, v in raw_products.items()}
        return {}

    def get_date_range(self) -> Dict[str, str]:
        """
        Identifies the minimum and maximum date values.
        """
        date_range = {"min": None, "max": None}
        if 'date' in self.df.columns:
            try:
                dates = pd.to_datetime(self.df['date'], errors='coerce')
                min_date = dates.min()
                max_date = dates.max()
                
                if pd.notnull(min_date):
                    date_range["min"] = min_date.strftime('%Y-%m-%d')
                if pd.notnull(max_date):
                    date_range["max"] = max_date.strftime('%Y-%m-%d')
            except Exception:
                min_val = self.df['date'].min()
                max_val = self.df['date'].max()
                date_range["min"] = str(min_val) if pd.notnull(min_val) else None
                date_range["max"] = str(max_val) if pd.notnull(max_val) else None
        return date_range

    def summarize(self) -> Dict[str, Any]:
        """
        Returns a complete summary dictionary of the data metrics.
        """
        if self.df.empty:
            return {
                "total_rows": 0,
                "total_files": 0,
                "revenue_by_country": {},
                "top_products": {},
                "date_range": {"min": None, "max": None}
            }
            
        return {
            "total_rows": self.get_total_rows(),
            "total_files": self.get_total_files(),
            "revenue_by_country": self.get_revenue_by_country(),
            "top_products": self.get_top_products(),
            "date_range": self.get_date_range()
        }

    def _find_column(self, candidates: List[str], default: str) -> str:
        """
        Helper method to identify a column name from potential candidate list.
        """
        for col in candidates:
            if col in self.df.columns:
                return col
        return default
