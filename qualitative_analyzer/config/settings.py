"""Configuration management for the qualitative analyzer."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    """Simplified configuration settings for the qualitative analyzer."""
    
    # Core Azure OpenAI (required)
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment_name: str = "gpt-4o-2"
    azure_openai_embedding_deployment_name: str = "text-embedding-ada-002"
    azure_api_version: str = "2024-10-21"
    
    # Essential analysis settings
    batch_size: int = 25
    max_tokens: int = 20000
    similarity_threshold: float = 0.85
    max_themes_per_text: int = 3
    
    # Theme consolidation settings
    final_theme_count: int = 10
    consolidation_deployment: str = "gpt-4o"  # Use GPT-4 for consolidation
    consolidation_temperature: float = 0.3  # Lower temperature for consistent consolidation
    consolidation_max_tokens: int = 1000
    enable_theme_consolidation: bool = True
    
    # OpenAI client settings
    openai_max_retries: int = 3
    openai_retry_delay: int = 5
    openai_timeout: int = 60
    backoff_factor: float = 2.0
    backoff_max_tries: int = 3
    
    # Theme mapping settings  
    theme_mapping_temperature: float = 0.2
    theme_mapping_max_tokens: int = 1000
    theme_mapping_chunk_size: int = 20
    
    # Theme analysis settings
    theme_analysis_max_words: int = 1000
    max_theme_examples: int = 30
    max_representative_quotes: int = 5
    
    # Data processing settings
    min_text_length: int = 5
    data_encoding: str = "utf-8"
    encoding_fallbacks: list = None
    
    # UI/Display settings
    top_themes_display_count: int = 5
    theme_threshold_for_display: int = 10
    
    # Default columns and settings (for backward compatibility)
    default_text_column: str = "text"
    api_temperature: float = 0.5
    
    def __post_init__(self):
        """Initialize derived settings after object creation."""
        if self.encoding_fallbacks is None:
            self.encoding_fallbacks = [self.data_encoding, 'utf-8', 'latin-1', 'cp1252']
    
    @classmethod
    def from_env(cls, env_file: str = None) -> "Settings":
        """Create settings from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Debug: Print the loaded values
        print(f"DEBUG: FINAL_THEME_COUNT from env: {os.getenv('FINAL_THEME_COUNT', 'NOT_SET')}")
        print(f"DEBUG: CONSOLIDATION_MAX_TOKENS from env: {os.getenv('CONSOLIDATION_MAX_TOKENS', 'NOT_SET')}")
        print(f"DEBUG: CONSOLIDATION_DEPLOYMENT from env: {os.getenv('CONSOLIDATION_DEPLOYMENT', 'NOT_SET')}")
            
        return cls(
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-2"),
            azure_openai_embedding_deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),
            azure_api_version=os.getenv("AZURE_API_VERSION", "2024-10-21"),
            batch_size=int(os.getenv("BATCH_SIZE", "25")),
            max_tokens=int(os.getenv("MAX_TOKENS", "20000")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
            max_themes_per_text=int(os.getenv("MAX_THEMES_PER_TEXT", "3")),
            final_theme_count=int(os.getenv("FINAL_THEME_COUNT", "10")),
            consolidation_deployment=os.getenv("CONSOLIDATION_DEPLOYMENT", "gpt-4o"),
            consolidation_temperature=float(os.getenv("CONSOLIDATION_TEMPERATURE", "0.3")),
            consolidation_max_tokens=int(os.getenv("CONSOLIDATION_MAX_TOKENS", "1000")),
            enable_theme_consolidation=os.getenv("ENABLE_THEME_CONSOLIDATION", "true").lower() == "true",
            openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            openai_retry_delay=int(os.getenv("OPENAI_RETRY_DELAY", "5")),
            openai_timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
            backoff_factor=float(os.getenv("BACKOFF_FACTOR", "2.0")),
            backoff_max_tries=int(os.getenv("BACKOFF_MAX_TRIES", "3")),
            theme_mapping_temperature=float(os.getenv("THEME_MAPPING_TEMPERATURE", "0.2")),
            theme_mapping_max_tokens=int(os.getenv("THEME_MAPPING_MAX_TOKENS", "1000")),
            theme_mapping_chunk_size=int(os.getenv("THEME_MAPPING_CHUNK_SIZE", "20")),
            theme_analysis_max_words=int(os.getenv("THEME_ANALYSIS_MAX_WORDS", "1000")),
            max_theme_examples=int(os.getenv("MAX_THEME_EXAMPLES", "30")),
            max_representative_quotes=int(os.getenv("MAX_REPRESENTATIVE_QUOTES", "5")),
            min_text_length=int(os.getenv("MIN_TEXT_LENGTH", "5")),
            data_encoding=os.getenv("DATA_ENCODING", "utf-8"),
            top_themes_display_count=int(os.getenv("TOP_THEMES_DISPLAY_COUNT", "5")),
            theme_threshold_for_display=int(os.getenv("THEME_THRESHOLD_FOR_DISPLAY", "10")),
            default_text_column=os.getenv("DEFAULT_TEXT_COLUMN", "text"),
            api_temperature=float(os.getenv("API_TEMPERATURE", "0.5"))
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required")
        
        if not self.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required")