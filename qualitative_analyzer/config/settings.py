"""Configuration management for the qualitative analyzer."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class Settings:
    """Configuration settings for the qualitative analyzer."""
    
    # Azure OpenAI Configuration
    azure_openai_deployment_name: str
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_embedding_deployment_name: str
    azure_api_version: str
    
    # OpenAI Fallback
    openai_api_key: Optional[str] = None
    
    # Analysis Configuration
    default_batch_size: int = 15
    max_tokens: int = 4000
    api_temperature: float = 0.5
    api_retries: int = 3
    api_retry_delay: int = 5
    
    # File Configuration
    default_text_column: str = "text"
    default_output_dir: str = "output"
    default_theme_column: str = "theme"
    default_summary_column: str = "summary"
    
    # Theme Analysis Configuration
    max_themes_per_entry: int = 3  # Original AI extraction limit
    max_themes_per_entry_consolidated: int = 5  # Per-entry limit for consolidated themes
    global_theme_limit: int = 15  # Maximum total unique consolidated themes
    theme_confidence_threshold: float = 0.6  # Minimum confidence for consolidated themes
    max_themes_in_report: int = 10
    max_quotes_per_theme: int = 5
    theme_similarity_threshold: float = 0.85
    enable_theme_similarity: bool = True
    enable_caching: bool = True
    
    # Theme Consolidation Configuration
    representative_themes_count: int = 10  # Number of representative themes to create
    theme_mapping_chunk_size: int = 25  # Chunk size for theme mapping
    consolidation_temperature_representative: float = 0.3  # Temperature for creating representative themes
    consolidation_temperature_mapping: float = 0.1  # Temperature for mapping themes
    consolidation_max_tokens_representative: int = 300  # Max tokens for representative theme creation
    consolidation_max_tokens_mapping: int = 800  # Max tokens for theme mapping
    
    # Report Configuration
    generate_theme_report: bool = True
    generate_theme_chart: bool = True
    max_themes_in_chart: int = 15
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "qualitative_analysis.log"
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Settings":
        """Create settings from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
            
        return cls(
            azure_openai_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-2"),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_embedding_deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),
            azure_api_version=os.getenv("AZURE_API_VERSION", "2024-10-21"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            default_batch_size=int(os.getenv("DEFAULT_BATCH_SIZE", "15")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
            api_temperature=float(os.getenv("API_TEMPERATURE", "0.5")),
            api_retries=int(os.getenv("API_RETRIES", "3")),
            api_retry_delay=int(os.getenv("API_RETRY_DELAY", "5")),
            default_text_column=os.getenv("DEFAULT_TEXT_COLUMN", "text"),
            default_output_dir=os.getenv("DEFAULT_OUTPUT_DIR", "output"),
            default_theme_column=os.getenv("DEFAULT_THEME_COLUMN", "theme"),
            default_summary_column=os.getenv("DEFAULT_SUMMARY_COLUMN", "summary"),
            max_themes_per_entry=int(os.getenv("MAX_THEMES_PER_ENTRY", "3")),
            max_themes_per_entry_consolidated=int(os.getenv("MAX_THEMES_PER_ENTRY_CONSOLIDATED", "5")),
            global_theme_limit=int(os.getenv("GLOBAL_THEME_LIMIT", "10")),
            theme_confidence_threshold=float(os.getenv("THEME_CONFIDENCE_THRESHOLD", "0.6")),
            max_themes_in_report=int(os.getenv("MAX_THEMES_IN_REPORT", "10")),
            max_quotes_per_theme=int(os.getenv("MAX_QUOTES_PER_THEME", "5")),
            theme_similarity_threshold=float(os.getenv("THEME_SIMILARITY_THRESHOLD", "0.85")),
            enable_theme_similarity=os.getenv("ENABLE_THEME_SIMILARITY", "true").lower() == "true",
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            representative_themes_count=int(os.getenv("REPRESENTATIVE_THEMES_COUNT", "10")),
            theme_mapping_chunk_size=int(os.getenv("THEME_MAPPING_CHUNK_SIZE", "25")),
            consolidation_temperature_representative=float(os.getenv("CONSOLIDATION_TEMPERATURE_REPRESENTATIVE", "0.3")),
            consolidation_temperature_mapping=float(os.getenv("CONSOLIDATION_TEMPERATURE_MAPPING", "0.1")),
            consolidation_max_tokens_representative=int(os.getenv("CONSOLIDATION_MAX_TOKENS_REPRESENTATIVE", "300")),
            consolidation_max_tokens_mapping=int(os.getenv("CONSOLIDATION_MAX_TOKENS_MAPPING", "800")),
            generate_theme_report=os.getenv("GENERATE_THEME_REPORT", "true").lower() == "true",
            generate_theme_chart=os.getenv("GENERATE_THEME_CHART", "true").lower() == "true",
            max_themes_in_chart=int(os.getenv("MAX_THEMES_IN_CHART", "15")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "qualitative_analysis.log")
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required")
        
        if not self.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required")
            
        if self.default_batch_size <= 0:
            raise ValueError("DEFAULT_BATCH_SIZE must be positive")
            
        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be positive")
            
        if not (0 <= self.api_temperature <= 2):
            raise ValueError("API_TEMPERATURE must be between 0 and 2")
            
        if not (0 <= self.theme_similarity_threshold <= 1):
            raise ValueError("THEME_SIMILARITY_THRESHOLD must be between 0 and 1")
            
        if self.max_themes_per_entry_consolidated <= 0:
            raise ValueError("MAX_THEMES_PER_ENTRY_CONSOLIDATED must be positive")
            
        if self.global_theme_limit <= 0:
            raise ValueError("GLOBAL_THEME_LIMIT must be positive")
            
        if not (0 <= self.theme_confidence_threshold <= 1):
            raise ValueError("THEME_CONFIDENCE_THRESHOLD must be between 0 and 1")
            
        if self.representative_themes_count <= 0:
            raise ValueError("REPRESENTATIVE_THEMES_COUNT must be positive")
            
        if self.theme_mapping_chunk_size <= 0:
            raise ValueError("THEME_MAPPING_CHUNK_SIZE must be positive")
            
        if not (0 <= self.consolidation_temperature_representative <= 2):
            raise ValueError("CONSOLIDATION_TEMPERATURE_REPRESENTATIVE must be between 0 and 2")
            
        if not (0 <= self.consolidation_temperature_mapping <= 2):
            raise ValueError("CONSOLIDATION_TEMPERATURE_MAPPING must be between 0 and 2")
            
        if self.consolidation_max_tokens_representative <= 0:
            raise ValueError("CONSOLIDATION_MAX_TOKENS_REPRESENTATIVE must be positive")
            
        if self.consolidation_max_tokens_mapping <= 0:
            raise ValueError("CONSOLIDATION_MAX_TOKENS_MAPPING must be positive")