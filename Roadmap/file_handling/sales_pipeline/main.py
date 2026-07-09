import io
import pathlib
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
import pandas as pd

from cleaner import clean_dataframe
from analyzer import summarize

app = FastAPI(
    title="Multi-File Sales Data Pipeline API",
    description="An API to upload regional sales CSVs, clean and combine them, and fetch summary metrics.",
    version="1.0.0"
)

# Directories management
BASE_DIR = pathlib.Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/upload", summary="Upload multiple regional CSV files")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accepts multiple CSV files, validates that each file is a CSV and is not empty,
    and saves them to the uploads directory.
    """
    try:
        # Create uploads folder if it doesn't exist
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for file in files:
            # 1. Validate file extension
            if not file.filename.endswith('.csv'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' is not a CSV. Only '.csv' files are accepted."
                )
            
            # Read file contents in-memory
            contents = await file.read()
            
            # Check if file is empty
            if not contents or len(contents.strip()) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' contains no rows/data."
                )
            
            # 2. Check if CSV file is readable and has rows
            try:
                df = pd.read_csv(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read CSV '{file.filename}': {str(e)}"
                )
                
            if df.empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' contains no rows/data."
                )
            
            # Save raw file to disk
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as f:
                f.write(contents)
                
            results.append({
                "filename": file.filename,
                "rows_received": len(df)
            })
            
        return {
            "message": f"Successfully uploaded {len(files)} file(s).",
            "files": results
        }
        
    except HTTPException as he:
        raise he
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
        # Check if uploads directory exists and contains CSV files
        if not UPLOAD_DIR.exists() or not any(UPLOAD_DIR.glob("*.csv")):
            raise HTTPException(
                status_code=400,
                detail="No CSV files found in uploads folder to process. Please upload files first."
            )
            
        cleaned_dfs = []
        total_rows_before = 0
        
        # Read and clean each CSV
        for file_path in UPLOAD_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read file '{file_path.name}': {str(e)}"
                )
                
            total_rows_before += len(df)
            
            # Clean dataframe using clean_dataframe logic
            try:
                cleaned_df = clean_dataframe(df)
            except ValueError as ve:
                raise HTTPException(
                    status_code=400,
                    detail=f"Validation error in file '{file_path.name}': {str(ve)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error cleaning file '{file_path.name}': {str(e)}"
                )
                
            # Add source_file column
            cleaned_df['source_file'] = file_path.name
            cleaned_dfs.append(cleaned_df)
            
        if not cleaned_dfs:
            raise HTTPException(
                status_code=400,
                detail="No valid data was extracted from the uploaded CSV files."
            )
            
        # Combine all dataframes
        combined_df = pd.concat(cleaned_dfs, ignore_index=True)
        
        # Save to outputs directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        combined_file_path = OUTPUT_DIR / "combined_sales.csv"
        combined_df.to_csv(combined_file_path, index=False)
        
        total_rows_after = len(combined_df)
        
        return {
            "message": "Files processed and combined successfully.",
            "total_rows_before": total_rows_before,
            "total_rows_after": total_rows_after,
            "combined_file": combined_file_path.name
        }
        
    except HTTPException as he:
        raise he
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
    combined_file = OUTPUT_DIR / "combined_sales.csv"
    try:
        if not combined_file.exists():
            raise FileNotFoundError()
            
        df = pd.read_csv(combined_file)
        summary = summarize(df)
        return summary
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Combined sales file not found. Please run the /process endpoint first."
        )
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
    combined_file = OUTPUT_DIR / "combined_sales.csv"
    try:
        if not combined_file.exists():
            raise FileNotFoundError()
            
        return FileResponse(
            path=combined_file,
            filename="combined_sales.csv",
            media_type="text/csv"
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Combined sales file not found. Please run the /process endpoint first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while preparing download: {str(e)}"
        )
