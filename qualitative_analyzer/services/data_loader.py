"""Data loading and preprocessing utilities."""

import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..utils.validators import validate_input_file

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of input data files."""
    
    def __init__(self, settings=None):
        """Initialize data loader with settings."""
        self.settings = settings
        self.encoding = settings.data_encoding if settings else 'utf-8'
        self.supported_formats = ['.xlsx', '.xls', '.csv', '.tsv']
    
    def load_data(
        self,
        file_path: str,
        text_column: str,
        required_columns: Optional[List[str]] = None,
        sheet_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from file with validation.
        
        Args:
            file_path: Path to input file
            text_column: Name of column containing text to analyze
            required_columns: List of required columns (defaults to [text_column])
            sheet_name: Specific sheet name for Excel files
            
        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        if required_columns is None:
            required_columns = [text_column]
        
        # Validate file
        is_valid, error_msg = validate_input_file(file_path, required_columns)
        if not is_valid:
            raise ValueError(f"File validation failed: {error_msg}")
        
        # Load based on file extension
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        
        try:
            if file_ext == '.csv':
                df = self._load_csv(file_path)
            elif file_ext == '.tsv':
                df = self._load_tsv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = self._load_excel(file_path, sheet_name)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise
        
        # Simple data cleaning
        df_cleaned = self._clean_data(df, text_column)
        
        # Generate simple metadata
        metadata = {
            "file_path": file_path,
            "file_size_mb": round(file_path_obj.stat().st_size / (1024 * 1024), 2),
            "original_row_count": len(df),
            "cleaned_row_count": len(df_cleaned),
            "columns": list(df.columns),
            "text_column": text_column,
            "cleaning_summary": {
                "rows_removed": len(df) - len(df_cleaned),
                "duplicates_removed": len(df) - len(df.drop_duplicates()),
                "empty_texts_removed": df[text_column].isnull().sum() + (df[text_column].str.strip() == "").sum()
            }
        }
        
        logger.info(f"Loaded {len(df_cleaned)} rows from {file_path}")
        return df_cleaned, metadata
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with encoding detection."""
        encodings_to_try = self.settings.encoding_fallbacks if self.settings else [self.encoding, 'utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded CSV with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Failed to load CSV with {encoding}: {str(e)}")
                break
        
        raise ValueError(f"Unable to load CSV file with any of the tried encodings: {encodings_to_try}")
    
    def _load_tsv(self, file_path: str) -> pd.DataFrame:
        """Load TSV file."""
        encodings_to_try = self.settings.encoding_fallbacks if self.settings else [self.encoding, 'utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, sep='\\t', encoding=encoding)
                logger.info(f"Successfully loaded TSV with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Failed to load TSV with {encoding}: {str(e)}")
                break
        
        raise ValueError(f"Unable to load TSV file with any of the tried encodings: {encodings_to_try}")
    
    def _load_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load Excel file."""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            logger.info(f"Successfully loaded Excel file")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Excel file: {str(e)}")
            raise
    
    def _clean_data(
        self,
        df: pd.DataFrame,
        text_column: str,
        min_text_length: int = None,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """Clean and preprocess the data."""
        if min_text_length is None:
            min_text_length = self.settings.min_text_length if self.settings else 5
            
        df_clean = df.copy()
        
        # Remove rows where text column is null or empty
        df_clean = df_clean.dropna(subset=[text_column])
        df_clean = df_clean[df_clean[text_column].str.strip() != ""]
        
        # Remove texts that are too short
        df_clean = df_clean[df_clean[text_column].str.len() >= min_text_length]
        
        # Clean text content
        df_clean[text_column] = df_clean[text_column].str.strip()
        
        # Remove duplicates in text column if requested
        if remove_duplicates:
            df_clean = df_clean.drop_duplicates(subset=[text_column])
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        return df_clean
    
    
    def export_cleaned_data(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = "xlsx"
    ) -> bool:
        """
        Export cleaned data to file.
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            format: Export format ('xlsx', 'csv')
            
        Returns:
            Success status
        """
        try:
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif format.lower() == 'xlsx':
                df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported {len(df)} rows to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export data to {output_path}: {str(e)}")
            return False