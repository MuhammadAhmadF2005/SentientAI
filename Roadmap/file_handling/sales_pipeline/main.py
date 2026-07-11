import io
import pathlib
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
import pandas as pd

from cleaner import DataCleaner
from analyzer import DataAnalyzer

class PipelineManager:
    """
    Manages file inputs, directories, and orchestrates cleaning and merging.
    """
    def __init__(self, upload_dir: pathlib.Path, output_dir: pathlib.Path):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.cleaner = DataCleaner()

    def save_upload(self, filename: str, contents: bytes) -> int:
        """
        Validates the format and content of an uploaded file,
        saves the raw file to disk, and returns the number of rows.
        """
        # Validate format
        if not filename.endswith('.csv'):
            raise ValueError(f"File '{filename}' is not a CSV. Only '.csv' files are accepted.")

        # Check empty bytes
        if not contents or len(contents.strip()) == 0:
            raise ValueError(f"File '{filename}' contains no rows/data.")

        # Parse CSV to check validity and row count
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise ValueError(f"Failed to read CSV '{filename}': {str(e)}")

        if df.empty:
            raise ValueError(f"File '{filename}' contains no rows/data.")

        # Save validated file contents to disk
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.upload_dir / filename
        with open(file_path, "wb") as f:
            f.write(contents)

        return len(df)

    def process_uploads(self) -> Dict[str, Any]:
        """
        Reads, cleans, and merges all CSVs in the uploads directory.
        Saves the results and returns metadata of the merged rows.
        """
        if not self.upload_dir.exists() or not any(self.upload_dir.glob("*.csv")):
            raise FileNotFoundError("No CSV files found in uploads folder to process. Please upload files first.")

        cleaned_dfs = []
        total_rows_before = 0

        for file_path in self.upload_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                raise ValueError(f"Failed to read file '{file_path.name}': {str(e)}")

            total_rows_before += len(df)

            try:
                cleaned_df = self.cleaner.clean(df)
            except ValueError as ve:
                raise ValueError(f"Validation error in file '{file_path.name}': {str(ve)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error cleaning file '{file_path.name}': {str(e)}")

            cleaned_df['source_file'] = file_path.name
            cleaned_dfs.append(cleaned_df)

        if not cleaned_dfs:
            raise ValueError("No valid data was extracted from the uploaded CSV files.")

        # Merge DataFrames
        combined_df = pd.concat(cleaned_dfs, ignore_index=True)

        # Export combined file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        combined_file_path = self.output_dir / "combined_sales.csv"
        combined_df.to_csv(combined_file_path, index=False)

        return {
            "total_rows_before": total_rows_before,
            "total_rows_after": len(combined_df),
            "combined_file": combined_file_path.name
        }

    def get_combined_file_path(self) -> pathlib.Path:
        """
        Retrieves the path of the combined sales CSV, raising FileNotFoundError if missing.
        """
        combined_file_path = self.output_dir / "combined_sales.csv"
        if not combined_file_path.exists():
            raise FileNotFoundError("Combined sales file not found. Please run the /process endpoint first.")
        return combined_file_path


# API Initialization
app = FastAPI(
    title="Multi-File Sales Data Pipeline API",
    description="An API to upload regional sales CSVs, clean and combine them, and fetch summary metrics using OOP principles.",
    version="1.1.0"
)

# Directory Setup
BASE_DIR = pathlib.Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Global Pipeline Manager Service Instance
manager = PipelineManager(upload_dir=UPLOAD_DIR, output_dir=OUTPUT_DIR)


@app.get("/", include_in_schema=False)
async def root():
    """Redirects base path visits to Interactive Swagger Docs."""
    return RedirectResponse(url="/docs")


@app.post("/upload", summary="Upload multiple regional CSV files")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accepts multiple CSV files, validates that each file is a CSV and is not empty,
    and saves them to the uploads directory.
    """
    try:
        results = []
        for file in files:
            contents = await file.read()
            rows = manager.save_upload(file.filename, contents)
            results.append({
                "filename": file.filename,
                "rows_received": rows
            })
        return {
            "message": f"Successfully uploaded {len(files)} file(s).",
            "files": results
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during upload: {str(e)}"
        )


@app.post("/process", summary="Clean and combine uploaded CSVs")
async def process_files():
    """
    Reads all CSVs from the uploads directory, cleans them, adds a source_file column,
    combines them, and saves the result to outputs/combined_sales.csv.
    """
    try:
        result = manager.process_uploads()
        return {
            "message": "Files processed and combined successfully.",
            **result
        }
    except (FileNotFoundError, ValueError) as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during processing: {str(e)}"
        )


@app.get("/summary", summary="Get aggregated metrics from combined sales data")
async def get_summary():
    """
    Reads the combined sales data and returns basic aggregation summary metrics.
    """
    try:
        combined_file = manager.get_combined_file_path()
        df = pd.read_csv(combined_file)
        analyzer = DataAnalyzer(df)
        return analyzer.summarize()
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while generating summary: {str(e)}"
        )


@app.get("/download", summary="Download the combined cleaned CSV file")
async def download_combined_file():
    """
    Downloads the final combined, cleaned CSV file.
    """
    try:
        combined_file = manager.get_combined_file_path()
        return FileResponse(
            path=combined_file,
            filename="combined_sales.csv",
            media_type="text/csv"
        )
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while preparing download: {str(e)}"
        )
