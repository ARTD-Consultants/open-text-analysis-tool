"""Simple validation utilities."""

import os
import pandas as pd
from typing import List, Tuple
from pathlib import Path


def validate_input_file(file_path: str, required_columns: List[str]) -> Tuple[bool, str]:
    """
    Validate input file exists and has required columns.
    
    Args:
        file_path: Path to the input file
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, f"Input file not found: {file_path}"
    
    # Check file extension
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in ['.xlsx', '.xls', '.csv']:
        return False, f"Unsupported file format: {file_ext}. Use .xlsx, .xls, or .csv"
    
    try:
        # Read file to check structure
        if file_ext == '.csv':
            df = pd.read_csv(file_path, nrows=1)
        else:
            df = pd.read_excel(file_path, nrows=1)
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            available_cols = list(df.columns)
            return False, (f"Missing required columns: {missing_columns}. "
                          f"Available columns: {available_cols}")
        
        return True, "File validation successful"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def validate_required_settings(api_key: str, endpoint: str, deployment: str) -> List[str]:
    """Validate required Azure OpenAI settings."""
    errors = []
    
    if not api_key:
        errors.append("AZURE_OPENAI_API_KEY is required")
    
    if not endpoint:
        errors.append("AZURE_OPENAI_ENDPOINT is required")
        
    if not deployment:
        errors.append("AZURE_OPENAI_DEPLOYMENT_NAME is required")
    
    return errors