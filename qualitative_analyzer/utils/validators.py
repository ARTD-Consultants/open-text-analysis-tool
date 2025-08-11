"""Input validation and data quality checking utilities."""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
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
            df = pd.read_csv(file_path, nrows=1)  # Read just first row for validation
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


def validate_configuration(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required configuration keys
    required_keys = [
        'azure_openai_api_key',
        'azure_openai_endpoint',
        'azure_openai_deployment_name'
    ]
    
    for key in required_keys:
        if key not in config or not config[key]:
            errors.append(f"Missing or empty required configuration: {key}")
    
    # Validate numeric configurations
    numeric_validations = [
        ('default_batch_size', 1, 100, int),
        ('max_tokens', 100, 32000, int),
        ('api_temperature', 0.0, 2.0, float),
        ('api_retries', 1, 10, int),
        ('max_themes_per_entry', 1, 10, int),
        ('theme_similarity_threshold', 0.0, 1.0, float)
    ]
    
    for key, min_val, max_val, expected_type in numeric_validations:
        if key in config:
            try:
                value = expected_type(config[key])
                if not (min_val <= value <= max_val):
                    errors.append(f"{key} must be between {min_val} and {max_val}, got {value}")
            except (ValueError, TypeError):
                errors.append(f"{key} must be a valid {expected_type.__name__}")
    
    # Validate boolean configurations
    boolean_keys = [
        'enable_theme_similarity',
        'enable_caching',
        'generate_theme_report',
        'generate_theme_chart'
    ]
    
    for key in boolean_keys:
        if key in config and not isinstance(config[key], bool):
            errors.append(f"{key} must be a boolean value")
    
    # Validate file paths
    if 'default_output_dir' in config:
        output_dir = config['default_output_dir']
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory '{output_dir}': {str(e)}")
    
    return len(errors) == 0, errors


def validate_text_data(
    df: pd.DataFrame, 
    text_column: str,
    min_text_length: int = 5,
    max_text_length: int = 10000
) -> Dict[str, Any]:
    """
    Validate text data quality.
    
    Args:
        df: DataFrame containing text data
        text_column: Name of the text column
        min_text_length: Minimum acceptable text length
        max_text_length: Maximum acceptable text length
        
    Returns:
        Dictionary with validation results and statistics
    """
    if text_column not in df.columns:
        return {
            "is_valid": False,
            "error": f"Text column '{text_column}' not found in data"
        }
    
    text_series = df[text_column]
    total_entries = len(text_series)
    
    # Count missing/empty values
    missing_count = text_series.isnull().sum()
    empty_count = (text_series.str.strip() == "").sum()
    
    # Check text lengths
    text_lengths = text_series.str.len().dropna()
    
    if len(text_lengths) == 0:
        return {
            "is_valid": False,
            "error": "No valid text entries found"
        }
    
    # Count texts outside acceptable length range
    too_short = (text_lengths < min_text_length).sum()
    too_long = (text_lengths > max_text_length).sum()
    
    # Calculate statistics
    valid_entries = total_entries - missing_count - empty_count - too_short - too_long
    valid_percentage = (valid_entries / total_entries) * 100
    
    # Detect potential encoding issues
    encoding_issues = 0
    sample_size = min(100, len(text_series))
    for text in text_series.sample(sample_size).dropna():
        if isinstance(text, str):
            # Check for common encoding issue patterns
            if '�' in text or '\\x' in text:
                encoding_issues += 1
    
    estimated_encoding_issues = int((encoding_issues / sample_size) * total_entries)
    
    # Calculate text quality score
    quality_score = max(0, min(100, valid_percentage - (estimated_encoding_issues / total_entries * 100)))
    
    return {
        "is_valid": valid_entries > 0,
        "total_entries": total_entries,
        "valid_entries": valid_entries,
        "missing_entries": missing_count,
        "empty_entries": empty_count,
        "too_short_entries": too_short,
        "too_long_entries": too_long,
        "valid_percentage": round(valid_percentage, 2),
        "estimated_encoding_issues": estimated_encoding_issues,
        "quality_score": round(quality_score, 2),
        "text_length_stats": {
            "mean": text_lengths.mean(),
            "median": text_lengths.median(),
            "min": text_lengths.min(),
            "max": text_lengths.max(),
            "std": text_lengths.std()
        },
        "recommendations": _generate_data_recommendations(
            valid_percentage, too_short, too_long, estimated_encoding_issues, total_entries
        )
    }


def _generate_data_recommendations(
    valid_percentage: float,
    too_short: int,
    too_long: int,
    encoding_issues: int,
    total_entries: int
) -> List[str]:
    """Generate data quality recommendations."""
    recommendations = []
    
    if valid_percentage < 70:
        recommendations.append("Consider improving data quality - less than 70% of entries are valid")
    
    if too_short > total_entries * 0.1:
        recommendations.append("Many entries are too short - consider lowering minimum length or filtering")
    
    if too_long > total_entries * 0.05:
        recommendations.append("Some entries are very long - consider truncation or separate processing")
    
    if encoding_issues > total_entries * 0.05:
        recommendations.append("Potential encoding issues detected - check file encoding")
    
    if valid_percentage > 90:
        recommendations.append("Excellent data quality - ready for analysis")
    elif valid_percentage > 80:
        recommendations.append("Good data quality - minor cleanup may be beneficial")
    
    return recommendations


def validate_analysis_results(
    results: List[Dict[str, Any]],
    expected_themes_per_entry: int = 3
) -> Dict[str, Any]:
    """
    Validate analysis results for quality and completeness.
    
    Args:
        results: List of analysis result dictionaries
        expected_themes_per_entry: Expected number of themes per entry
        
    Returns:
        Dictionary with validation results
    """
    if not results:
        return {
            "is_valid": False,
            "error": "No analysis results provided"
        }
    
    total_results = len(results)
    
    # Count results with missing data
    missing_summary = sum(1 for r in results if not r.get('summary', '').strip())
    missing_themes = sum(1 for r in results if not r.get('themes', []))
    
    # Analyze theme distribution
    all_themes = []
    themes_per_entry = []
    
    for result in results:
        themes = result.get('themes', [])
        all_themes.extend(themes)
        themes_per_entry.append(len(themes))
    
    unique_themes = len(set(all_themes))
    avg_themes_per_entry = sum(themes_per_entry) / len(themes_per_entry) if themes_per_entry else 0
    
    # Calculate completeness percentage
    complete_results = total_results - missing_summary - missing_themes
    completeness_percentage = (complete_results / total_results) * 100
    
    # Check for potential issues
    issues = []
    if missing_summary > total_results * 0.1:
        issues.append(f"{missing_summary} entries missing summaries ({missing_summary/total_results*100:.1f}%)")
    
    if missing_themes > total_results * 0.05:
        issues.append(f"{missing_themes} entries missing themes ({missing_themes/total_results*100:.1f}%)")
    
    if unique_themes < 3:
        issues.append("Very few unique themes detected - may indicate analysis issues")
    
    if unique_themes > total_results * 0.8:
        issues.append("Too many unique themes - may need theme consolidation")
    
    return {
        "is_valid": complete_results > total_results * 0.8,
        "total_results": total_results,
        "complete_results": complete_results,
        "completeness_percentage": round(completeness_percentage, 2),
        "missing_summaries": missing_summary,
        "missing_themes": missing_themes,
        "unique_themes": unique_themes,
        "average_themes_per_entry": round(avg_themes_per_entry, 2),
        "theme_diversity_ratio": round(unique_themes / total_results, 3),
        "issues": issues,
        "quality_assessment": _assess_analysis_quality(completeness_percentage, unique_themes, total_results)
    }


def _assess_analysis_quality(
    completeness: float,
    unique_themes: int,
    total_results: int
) -> str:
    """Assess overall analysis quality."""
    if completeness >= 95 and 5 <= unique_themes <= total_results * 0.5:
        return "Excellent"
    elif completeness >= 85 and 3 <= unique_themes <= total_results * 0.7:
        return "Good"
    elif completeness >= 70:
        return "Fair"
    else:
        return "Poor"