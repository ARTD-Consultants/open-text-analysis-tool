"""Main qualitative analyzer orchestrating all components."""

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
from tqdm import tqdm
import re

from ..config.settings import Settings
from ..config.prompts import Prompts
from ..models.analysis_result import AnalysisResult, BatchResult, AnalysisSession
from ..models.batch import Batch, BatchConfig
from ..models.theme import ThemeHierarchy
from ..services.openai_client import OpenAIClient
from ..services.data_loader import DataLoader
from ..services.report_generator import ReportGenerator
from ..core.theme_manager import ThemeManager
from ..core.batch_processor import BatchProcessor
from ..core.cache_manager import CacheManager
from ..utils.token_counter import TokenCounter
from ..utils.similarity import SimilarityCalculator

logger = logging.getLogger(__name__)


class QualitativeAnalyzer:
    """Main analyzer class orchestrating all components."""
    
    def __init__(
        self,
        settings: Settings,
        enable_caching: bool = True,
        enable_theme_similarity: bool = True
    ):
        """Initialize the qualitative analyzer."""
        self.settings = settings
        self.settings.validate()
        
        # Initialize components
        self._initialize_components(enable_caching, enable_theme_similarity)
        
        # Analysis session tracking
        self.current_session: Optional[AnalysisSession] = None
        
        logger.info("QualitativeAnalyzer initialized successfully")
    
    @classmethod
    def from_env(
        cls, 
        env_file: Optional[str] = None,
        enable_caching: bool = True,
        enable_theme_similarity: bool = True
    ) -> "QualitativeAnalyzer":
        """Create analyzer from environment variables."""
        settings = Settings.from_env(env_file)
        return cls(settings, enable_caching, enable_theme_similarity)
    
    def _initialize_components(self, enable_caching: bool, enable_theme_similarity: bool):
        """Initialize all analyzer components."""
        # Core utilities
        self.token_counter = TokenCounter()
        
        # OpenAI client
        self.openai_client = OpenAIClient(
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_api_version,
            deployment_name=self.settings.azure_openai_deployment_name,
            embedding_deployment_name=self.settings.azure_openai_embedding_deployment_name,
            max_retries=self.settings.api_retries,
            retry_delay=self.settings.api_retry_delay
        )
        
        # Cache manager
        self.cache_manager = None
        if enable_caching and self.settings.enable_caching:
            self.cache_manager = CacheManager()
        
        # Similarity calculator with embeddings
        self.similarity_calculator = None
        if enable_theme_similarity and self.settings.enable_theme_similarity:
            self.similarity_calculator = SimilarityCalculator(
                embedding_client=self.openai_client,
                cache_dir="embeddings_cache" if enable_caching else None
            )
        
        # Theme manager
        self.theme_manager = ThemeManager(
            similarity_calculator=self.similarity_calculator,
            similarity_threshold=self.settings.theme_similarity_threshold,
            enable_validation=enable_theme_similarity,
            global_theme_limit=self.settings.global_theme_limit
        )
        
        # Batch processor
        batch_config = BatchConfig(
            max_batch_size=self.settings.default_batch_size,
            max_tokens_per_batch=self.settings.max_tokens,
            enable_dynamic_sizing=True
        )
        self.batch_processor = BatchProcessor(
            token_counter=self.token_counter,
            batch_config=batch_config
        )
        
        # Service components
        self.data_loader = DataLoader()
        self.report_generator = ReportGenerator(self.settings.default_output_dir)
    
    
    def analyze_file(
        self,
        input_file: str,
        text_column: str = None,
        output_file: str = None,
        sheet_name: str = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> AnalysisSession:
        """
        Analyze a file containing text data.
        
        Args:
            input_file: Path to input file (Excel, CSV)
            text_column: Column containing text to analyze
            output_file: Path for output file (optional)
            sheet_name: Excel sheet name (optional)
            progress_callback: Progress callback function
            
        Returns:
            AnalysisSession with complete results
        """
        logger.info(f"Starting analysis of file: {input_file}")
        
        # Use default column name if not specified
        if text_column is None:
            text_column = self.settings.default_text_column
        
        # Load and validate data
        df, metadata = self.data_loader.load_data(
            file_path=input_file,
            text_column=text_column,
            sheet_name=sheet_name
        )
        
        logger.info(f"Loaded {len(df)} valid text entries for analysis")
        
        # Start new analysis session
        self.current_session = AnalysisSession()
        self.current_session.total_entries = len(df)
        self.current_session.configuration = {
            "input_file": input_file,
            "text_column": text_column,
            "settings": self.settings.__dict__,
            "data_metadata": metadata
        }
        
        try:
            # Extract texts
            texts = df[text_column].tolist()
            
            # Create optimized batches
            batches = self.batch_processor.create_optimized_batches(texts)
            
            # Process batches (preserves original themes only)
            batch_results = self._process_text_batches(
                batches, 
                progress_callback=progress_callback
            )
            
            # Add results to session
            for batch_result in batch_results:
                self.current_session.add_batch(batch_result)
            
            # Apply consolidation to all results after processing
            all_results = self.current_session.get_all_results()
            self.theme_manager.apply_theme_consolidation(
                all_results, 
                self.openai_client, 
                self.settings
            )
            
            # Generate comprehensive results
            self._finalize_analysis_session()
            
            # Save results if output file specified
            if output_file:
                self._save_analysis_results(df, output_file)
            
            logger.info(f"Analysis completed successfully. "
                       f"Processed {self.current_session.total_entries} entries, "
                       f"identified {len(self.theme_manager.hierarchy.get_all_themes_flat())} unique themes")
            
            return self.current_session
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            if self.current_session:
                self.current_session.mark_completed()
            raise
    
    def _process_text_batches(
        self,
        batches: List[Batch],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchResult]:
        """Process batches of texts for analysis."""
        
        def process_single_batch(batch: Batch) -> BatchResult:
            """Process a single batch of texts."""
            batch_result = BatchResult(batch_id=batch.batch_id)
            
            try:
                # Get existing themes for consistency
                existing_themes = [
                    theme.name for theme in 
                    self.theme_manager.hierarchy.get_all_themes_flat()[:20]
                ]
                
                # Create prompt
                prompt = Prompts.batch_analysis_prompt(
                    batch.texts,
                    existing_themes,
                    self.settings.max_themes_per_entry
                )
                
                # Check cache first
                cached_response = None
                if self.cache_manager:
                    model_config = {
                        "temperature": self.settings.api_temperature,
                        "max_tokens": self.settings.max_tokens,
                        "model": self.settings.azure_openai_deployment_name
                    }
                    cached_response = self.cache_manager.get_api_response(prompt, model_config)
                
                # Get AI response
                if cached_response:
                    response = cached_response
                    logger.debug(f"Used cached response for batch {batch.batch_id}")
                else:
                    response = self.openai_client.analyze_text_batch(
                        prompt=prompt,
                        temperature=self.settings.api_temperature,
                        max_tokens=self.settings.max_tokens
                    )
                    
                    # Cache the response
                    if self.cache_manager:
                        self.cache_manager.cache_api_response(prompt, model_config, response)
                
                # Parse response into analysis results (preserves original themes)
                parsed_results = self._parse_batch_response(
                    response, 
                    batch.texts, 
                    len(batch.texts)
                )
                
                # Keep original themes intact - no consolidation during batch processing
                # Consolidation will happen after all batches are processed
                
                # Add results to batch
                for result in parsed_results:
                    batch_result.add_result(result)
                
                # Update statistics
                batch_result.tokens_used = self.token_counter.count_tokens(prompt + response)
                batch_result.api_calls = 1
                batch_result.mark_completed()
                
                logger.debug(f"Successfully processed batch {batch.batch_id} "
                           f"({len(batch.texts)} texts, {batch_result.tokens_used} tokens)")
                
                return batch_result
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch.batch_id}: {str(e)}")
                batch_result.errors.append(str(e))
                batch_result.mark_completed()
                batch.mark_failed(str(e))
                return batch_result
        
        # Process batches (sequential for now, can be made parallel)
        return self.batch_processor.process_batch_sequential(
            batches, 
            process_single_batch,
            progress_callback
        )
    
    def _parse_batch_response(
        self,
        response: str,
        original_texts: List[str],
        expected_count: int
    ) -> List[AnalysisResult]:
        """Parse AI response into structured analysis results."""
        results = []
        lines = response.split('\n')
        
        current_summary = None
        current_themes = None
        entry_index = 0
        
        for line in lines:
            line = line.strip()
            
            # Match entry pattern (e.g., "1:", "2:", etc.)
            entry_match = re.match(r'^(\d+):', line)
            if entry_match:
                # Save previous entry if exists
                if current_summary is not None and entry_index < len(original_texts):
                    result = self._create_analysis_result(
                        original_texts[entry_index], 
                        current_summary, 
                        current_themes
                    )
                    results.append(result)
                    entry_index += 1
                
                # Parse new entry
                entry_content = line.split(':', 1)[1].strip()
                if '|' in entry_content:
                    parts = entry_content.split('|', 1)
                    current_summary = parts[0].strip()
                    current_themes = parts[1].strip() if len(parts) > 1 else ""
                else:
                    current_summary = entry_content
                    current_themes = ""
                
                continue
            
            # Handle multi-line responses
            if line.startswith('Summary:'):
                current_summary = line.replace('Summary:', '').strip()
            elif line.startswith('Themes:'):
                current_themes = line.replace('Themes:', '').strip()
        
        # Add final entry
        if current_summary is not None and entry_index < len(original_texts):
            result = self._create_analysis_result(
                original_texts[entry_index], 
                current_summary, 
                current_themes
            )
            results.append(result)
        
        # Ensure we have the expected number of results
        while len(results) < expected_count and len(results) < len(original_texts):
            result = AnalysisResult(
                original_text=original_texts[len(results)],
                summary="Error: Analysis failed",
                themes=["Analysis Error"]
            )
            results.append(result)
        
        return results[:expected_count]
    
    def _create_analysis_result(
        self,
        text: str,
        summary: str,
        themes_str: str
    ) -> AnalysisResult:
        """Create an AnalysisResult from parsed components."""
        result = AnalysisResult(
            original_text=text,
            summary=summary
        )
        
        # Parse themes
        if themes_str:
            # Split by comma and clean
            theme_parts = [t.strip() for t in themes_str.split(',') if t.strip()]
            
            for theme_part in theme_parts:
                # Check for confidence scores in parentheses
                confidence_match = re.search(r'\(([0-9]+)%\)', theme_part)
                if confidence_match:
                    theme_name = re.sub(r'\([0-9]+%\)', '', theme_part).strip()
                    confidence = int(confidence_match.group(1)) / 100.0
                else:
                    theme_name = theme_part
                    confidence = 1.0
                
                if theme_name:
                    result.add_original_theme(theme_name, confidence)
        
        return result
    
    def _finalize_analysis_session(self):
        """Finalize the analysis session with comprehensive statistics."""
        if not self.current_session:
            return
        
        # Mark session as completed
        self.current_session.mark_completed()
        
        # Update session metadata
        self.current_session.session_metadata.update({
            "theme_statistics": self.theme_manager.get_theme_statistics(),
            "processing_statistics": self.batch_processor.get_processing_statistics(),
            "openai_statistics": self.openai_client.get_usage_statistics()
        })
        
        if self.cache_manager:
            self.current_session.session_metadata["cache_statistics"] = self.cache_manager.get_cache_statistics()
    
    def _save_analysis_results(self, original_df: pd.DataFrame, output_file: str):
        """Save analysis results to file."""
        if not self.current_session:
            return
        
        # Create results dataframe
        all_results = self.current_session.get_all_results()
        
        results_data = []
        for i, result in enumerate(all_results):
            row = original_df.iloc[i].to_dict() if i < len(original_df) else {}
            row.update({
                self.settings.default_summary_column: result.summary,
                self.settings.default_theme_column: ", ".join(result.themes),
                "original_themes": ", ".join(result.original_themes),
                "theme_count": len(result.themes),
                "original_theme_count": len(result.original_themes),
                "average_confidence": result.get_average_confidence(),
                "themes_filtered": len(result.original_themes) - len([t for t in result.themes if t != "NA"]),
                "processing_timestamp": result.processing_timestamp.isoformat()
            })
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save to file
        file_ext = os.path.splitext(output_file)[1].lower()
        if file_ext == '.csv':
            results_df.to_csv(output_file, index=False)
        else:
            results_df.to_excel(output_file, index=False)
        
        logger.info(f"Analysis results saved to {output_file}")
    
    def generate_comprehensive_report(
        self,
        report_title: str = "Qualitative Analysis Report"
    ) -> Dict[str, str]:
        """Generate comprehensive analysis report."""
        if not self.current_session:
            raise ValueError("No analysis session available for reporting")
        
        return self.report_generator.generate_comprehensive_report(
            session=self.current_session,
            theme_hierarchy=self.theme_manager.hierarchy,
            ai_client=self.openai_client,
            report_title=report_title
        )
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of current analysis."""
        if not self.current_session:
            return {"error": "No analysis session available"}
        
        return {
            "session_stats": self.current_session.get_session_statistics(),
            "theme_stats": self.theme_manager.get_theme_statistics(),
            "processing_stats": self.batch_processor.get_processing_statistics(),
            "top_themes": [
                {"name": theme.name, "frequency": theme.frequency, "confidence": theme.average_confidence}
                for theme in self.theme_manager.hierarchy.get_all_themes_flat()[:10]
            ]
        }
    
    def suggest_theme_merges(self, max_suggestions: int = 10) -> List[Dict[str, Any]]:
        """Get suggestions for theme merges."""
        suggestions = self.theme_manager.get_theme_suggestions(
            min_frequency=2,
            max_suggestions=max_suggestions
        )
        
        return [
            {
                "theme1": theme1,
                "theme2": theme2,
                "similarity": similarity,
                "recommendation": "merge" if similarity > 0.9 else "review"
            }
            for theme1, theme2, similarity in suggestions
        ]
    
    def test_connection(self) -> Dict[str, bool]:
        """Test connections to all external services."""
        return {
            "openai_connection": self.openai_client.test_connection(),
            "cache_available": self.cache_manager is not None,
            "embeddings_available": self.similarity_calculator is not None
        }