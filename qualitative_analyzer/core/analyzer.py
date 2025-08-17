"""Main qualitative analyzer orchestrating all components."""

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
from tqdm import tqdm
import re
from collections import Counter
from pathlib import Path

from ..config.settings import Settings
from ..config.prompts import Prompts
from ..models.analysis_result import AnalysisResult, BatchResult, AnalysisSession
# Simplified analyzer - no complex batch models needed
# Simplified analyzer - no complex theme hierarchies needed
from ..services.openai_client import OpenAIClient
from ..services.data_loader import DataLoader
from ..services.report_generator import ReportGenerator
from ..core.theme_manager import ThemeManager
from ..core.batch_processor import BatchProcessor
from ..core.cache_manager import SimpleCacheManager
from ..utils.token_counter import TokenCounter
# Simplified analyzer - no similarity calculator needed

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
            settings=self.settings
        )
        
        # Cache manager (simplified and optional)
        self.cache_manager = SimpleCacheManager(enabled=enable_caching)
        
        # Theme manager (simplified)
        self.theme_manager = ThemeManager(
            similarity_threshold=self.settings.similarity_threshold
        )
        
        # Batch processor (simplified)
        self.batch_processor = BatchProcessor(
            token_counter=self.token_counter,
            batch_size=self.settings.batch_size,
            max_tokens=self.settings.max_tokens
        )
        
        # Service components
        self.data_loader = DataLoader(settings=self.settings)
        self.report_generator = ReportGenerator("output")
    
    
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
            
            # Create simple batches
            batches = self.batch_processor.create_batches(texts)
            
            # Process batches
            batch_results = self._process_text_batches(
                batches, 
                progress_callback=progress_callback
            )
            
            # Add results to session
            for batch_result in batch_results:
                self.current_session.add_batch(batch_result)
            
            # Step 1: Apply basic theme consolidation (existing logic)
            all_results = self.current_session.get_all_results()
            self.theme_manager.apply_theme_consolidation(
                all_results, 
                self.openai_client, 
                self.settings
            )
            
            # Step 2: Advanced consolidation using GPT-4 if enabled
            if self.settings.enable_theme_consolidation:
                self._apply_advanced_theme_consolidation(all_results)
            
            # Generate comprehensive results
            self._finalize_analysis_session()
            
            # Save results if output file specified
            if output_file:
                self._save_analysis_results(df, output_file)
            
            theme_stats = self.theme_manager.get_theme_statistics()
            logger.info(f"Analysis completed successfully. "
                       f"Processed {self.current_session.total_entries} entries, "
                       f"identified {theme_stats['unique_themes']} unique themes")
            
            return self.current_session
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            if self.current_session:
                self.current_session.mark_completed()
            raise
    
    def _process_text_batches(
        self,
        batches: List[List[str]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchResult]:
        """Process batches of texts for analysis."""
        
        def process_single_batch(batch: List[str]) -> BatchResult:
            """Process a single batch of texts."""
            batch_result = BatchResult()
            
            try:
                # Get existing themes for consistency
                existing_themes = list(self.theme_manager.get_theme_summary().keys())[:20]
                
                # Create prompt
                prompt = Prompts.batch_analysis_prompt(
                    batch,
                    existing_themes,
                    self.settings.max_themes_per_text
                )
                
                # Check cache first
                cached_response = None
                if self.cache_manager:
                    cached_response = self.cache_manager.get_api_response(prompt)
                
                # Get AI response
                if cached_response:
                    response = cached_response
                    logger.debug(f"Used cached response for batch of {len(batch)} texts")
                else:
                    response = self.openai_client.analyze_text_batch(
                        prompt=prompt,
                        temperature=self.settings.api_temperature,
                        max_tokens=self.settings.max_tokens
                    )
                    
                    # Cache the response
                    if self.cache_manager:
                        self.cache_manager.cache_api_response(prompt, response)
                
                # Parse response into analysis results
                parsed_results = self._parse_batch_response(
                    response, 
                    batch, 
                    len(batch)
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
                
                logger.debug(f"Successfully processed batch ({len(batch)} texts, {batch_result.tokens_used} tokens)")
                
                return batch_result
                
            except Exception as e:
                logger.error(f"Failed to process batch: {str(e)}")
                batch_result.errors.append(str(e))
                batch_result.mark_completed()
                return batch_result
        
        # Process batches sequentially
        results = []
        for i, batch in enumerate(batches):
            result = process_single_batch(batch)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(batches))
        
        return results
    
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
        
        # Session statistics are already collected via get_session_statistics()
        # Additional statistics are available from theme_manager, batch_processor, and openai_client
        logger.debug("Analysis session finalized with statistics available")
        
    
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
                "summary": result.summary,
                "themes": ", ".join(result.themes),
                "original_themes": ", ".join(result.original_themes),
                "consolidated_themes": ", ".join(result.consolidated_themes),
                "theme_count": len(result.themes),
                "original_theme_count": len(result.original_themes),
                "consolidated_theme_count": len(result.consolidated_themes),
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
            # Save to Excel with multiple sheets including summary and charts
            output_path = Path(output_file)
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
                
                # Add summary sheet with statistics and charts
                self._create_summary_sheet(writer, all_results)
        
        logger.info(f"Analysis results saved to {output_file}")
    
    def generate_simple_report(
        self,
        report_title: str = "Qualitative Analysis Report"
    ) -> Dict[str, str]:
        """Generate simplified analysis report."""
        if not self.current_session:
            raise ValueError("No analysis session available for reporting")
        
        theme_summary = self.theme_manager.get_theme_summary()
        
        return self.report_generator.generate_simple_report(
            session=self.current_session,
            theme_summary=theme_summary,
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
                {"name": theme, "frequency": count}
                for theme, count in self.theme_manager.get_sorted_themes()[:10]
            ]
        }
    
    def suggest_theme_merges(self, max_suggestions: int = 10) -> List[Dict[str, Any]]:
        """Get basic theme suggestions (simplified)."""
        # Simple case-insensitive similarity suggestions
        theme_summary = self.theme_manager.get_theme_summary()
        suggestions = []
        
        themes = list(theme_summary.keys())
        for i, theme1 in enumerate(themes):
            for theme2 in themes[i+1:]:
                if theme1.lower() in theme2.lower() or theme2.lower() in theme1.lower():
                    suggestions.append({
                        "theme1": theme1,
                        "theme2": theme2,
                        "similarity": 0.8,  # Simple heuristic
                        "recommendation": "review"
                    })
                    
                if len(suggestions) >= max_suggestions:
                    break
            if len(suggestions) >= max_suggestions:
                break
        
        return suggestions
    
    def _apply_advanced_theme_consolidation(self, all_results: List[AnalysisResult]):
        """Apply advanced theme consolidation using GPT-4."""
        logger.info("Starting advanced theme consolidation using GPT-4...")
        
        # Collect all unique themes from the analysis
        all_themes = []
        for result in all_results:
            all_themes.extend(result.themes)
        
        # Remove duplicates while preserving order
        unique_themes = []
        seen = set()
        for theme in all_themes:
            if theme not in seen and theme != "NA":
                unique_themes.append(theme)
                seen.add(theme)
        
        if len(unique_themes) <= self.settings.final_theme_count:
            logger.info(f"Only {len(unique_themes)} unique themes found, skipping consolidation")
            # Just copy themes to consolidated themes
            for result in all_results:
                result.consolidated_themes = result.themes.copy()
            return
        
        logger.info(f"Consolidating {len(unique_themes)} themes into {self.settings.final_theme_count} consolidated themes")
        
        try:
            # Use GPT-4 to consolidate themes
            consolidated_theme_names = self.openai_client.consolidate_themes(
                original_themes=unique_themes,
                final_theme_count=self.settings.final_theme_count,
                consolidation_deployment=self.settings.consolidation_deployment
            )
            
            logger.info(f"Generated consolidated themes: {consolidated_theme_names}")
            
            # Map original themes to consolidated themes using similarity/GPT
            theme_mapping = self._create_theme_mapping(unique_themes, consolidated_theme_names)
            
            # Apply mapping to all results
            for result in all_results:
                result.consolidated_themes = []
                for original_theme in result.themes:
                    if original_theme in theme_mapping:
                        consolidated_theme = theme_mapping[original_theme]
                        if consolidated_theme not in result.consolidated_themes:
                            result.consolidated_themes.append(consolidated_theme)
                    elif original_theme != "NA":
                        # Fallback: keep original theme if no mapping found
                        if original_theme not in result.consolidated_themes:
                            result.consolidated_themes.append(original_theme)
            
            logger.info(f"Advanced theme consolidation completed successfully")
            
        except Exception as e:
            logger.error(f"Advanced theme consolidation failed: {str(e)}")
            # Fallback: copy original themes to consolidated themes
            for result in all_results:
                result.consolidated_themes = result.themes.copy()
    
    def _create_theme_mapping(self, original_themes: List[str], consolidated_themes: List[str]) -> Dict[str, str]:
        """Create mapping from original themes to consolidated themes."""
        mapping = {}
        
        # Process themes in chunks to avoid token limits
        chunk_size = self.settings.theme_mapping_chunk_size
        for i in range(0, len(original_themes), chunk_size):
            chunk = original_themes[i:i+chunk_size]
            
            try:
                # Use GPT to map this chunk
                prompt = Prompts.theme_mapping_prompt(chunk, consolidated_themes)
                response = self.openai_client.analyze_text_batch(
                    prompt=prompt,
                    temperature=self.settings.theme_mapping_temperature,
                    max_tokens=self.settings.theme_mapping_max_tokens
                )
                
                # Parse the mapping response
                for line in response.split('\n'):
                    line = line.strip()
                    if ' -> ' in line:
                        parts = line.split(' -> ', 1)
                        if len(parts) == 2:
                            original_theme = parts[0].strip()
                            consolidated_theme = parts[1].strip()
                            
                            # Validate that the consolidated theme exists
                            if consolidated_theme in consolidated_themes:
                                mapping[original_theme] = consolidated_theme
                            else:
                                # Find best match by partial string matching
                                best_match = None
                                for cons_theme in consolidated_themes:
                                    if consolidated_theme.lower() in cons_theme.lower() or cons_theme.lower() in consolidated_theme.lower():
                                        best_match = cons_theme
                                        break
                                
                                if best_match:
                                    mapping[original_theme] = best_match
                                else:
                                    # Fallback to first consolidated theme
                                    mapping[original_theme] = consolidated_themes[0]
                
            except Exception as e:
                logger.error(f"Failed to map theme chunk: {str(e)}")
                # Fallback mapping: assign to first consolidated theme
                for theme in chunk:
                    mapping[theme] = consolidated_themes[0] if consolidated_themes else "Uncategorized"
        
        return mapping
    
    def _create_summary_sheet(self, writer, all_results: List[AnalysisResult]):
        """Create comprehensive summary sheet with statistics and visualizations."""
        
        # Collect theme counts
        original_themes = []
        consolidated_themes = []
        
        for result in all_results:
            original_themes.extend(result.themes)
            if result.consolidated_themes:
                consolidated_themes.extend(result.consolidated_themes)
        
        # Remove NA themes
        original_themes = [t for t in original_themes if t != "NA"]
        consolidated_themes = [t for t in consolidated_themes if t != "NA"]
        
        # Count themes
        original_counts = Counter(original_themes)
        consolidated_counts = Counter(consolidated_themes)
        
        # Calculate token and batch statistics from current session
        session_stats = self.current_session.get_session_statistics() if self.current_session else {}
        total_tokens = session_stats.get('total_tokens_used', 0)
        total_batches = len(self.current_session.batches) if self.current_session else 0
        avg_tokens_per_batch = total_tokens / total_batches if total_batches > 0 else 0
        
        # Calculate batch size statistics
        batch_sizes = [len(batch.results) for batch in self.current_session.batches] if self.current_session else []
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
        configured_batch_size = self.settings.batch_size
        
        # Get prompts used for transparency
        from ..config.prompts import Prompts
        initial_prompt = Prompts.batch_analysis_prompt(["[sample text]"], max_themes=3).replace("\n", " | ")
        consolidation_prompt = Prompts.theme_consolidation_prompt(["[sample themes]"], final_theme_count=15).replace("\n", " | ")
        
        # Create summary statistics
        stats_data = [
            ["Analysis Summary", ""],
            ["Total Entries Processed", len(all_results)],
            ["", ""],
            ["Original Themes", ""],
            ["Total Original Theme Occurrences", len(original_themes)],
            ["Unique Original Themes", len(original_counts)],
            ["Most Common Original Theme", original_counts.most_common(1)[0] if original_counts else ("None", 0)],
            ["", ""],
            ["Consolidated Themes", ""],
            ["Total Consolidated Theme Occurrences", len(consolidated_themes)],
            ["Unique Consolidated Themes", len(consolidated_counts)],
            ["Most Common Consolidated Theme", consolidated_counts.most_common(1)[0] if consolidated_counts else ("None", 0)],
            ["", ""],
            ["Processing Statistics", ""],
            ["Average Themes per Entry (Original)", len(original_themes) / len(all_results) if all_results else 0],
            ["Average Themes per Entry (Consolidated)", len(consolidated_themes) / len(all_results) if all_results else 0],
            ["Theme Consolidation Ratio", f"{len(original_counts)} → {len(consolidated_counts)}" if consolidated_counts else "Not applied"],
            ["Total Tokens Used", total_tokens],
            ["Total Batches Processed", total_batches], 
            ["Average Tokens per Batch", f"{avg_tokens_per_batch:.0f}" if avg_tokens_per_batch > 0 else "N/A"],
            ["Configured Batch Size (.env)", configured_batch_size],
            ["Actual Average Batch Size", f"{avg_batch_size:.1f}" if avg_batch_size > 0 else "N/A"],
            ["Batch Sizes", ", ".join(map(str, batch_sizes)) if batch_sizes else "N/A"],
            ["", ""],
            ["Prompts Used", ""],
            ["Initial Theme Analysis Prompt", initial_prompt],
            ["Theme Consolidation Prompt", consolidation_prompt],
        ]
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=0, startcol=0)
        
        # Create original themes count table
        if original_counts:
            original_df = pd.DataFrame([
                (theme, count) for theme, count in original_counts.most_common()
            ], columns=['Original Theme', 'Count'])
            original_df.to_excel(writer, sheet_name='Summary', index=False, startrow=0, startcol=4)
        
        # Create consolidated themes count table
        if consolidated_counts:
            consolidated_df = pd.DataFrame([
                (theme, count) for theme, count in consolidated_counts.most_common()
            ], columns=['Consolidated Theme', 'Count'])
            consolidated_df.to_excel(writer, sheet_name='Summary', index=False, startrow=0, startcol=7)
        
        # Add charts
        self._add_theme_charts(writer, original_counts, consolidated_counts)
    
    def _add_theme_charts(self, writer, original_counts: Counter, consolidated_counts: Counter):
        """Add charts to the summary sheet."""
        try:
            from openpyxl.chart import BarChart, Reference
            from openpyxl.chart.label import DataLabelList
            
            worksheet = writer.sheets['Summary']
            
            # Original themes chart (top 15)
            if original_counts:
                top_original = dict(original_counts.most_common(15))
                
                # Create chart
                chart1 = BarChart()
                chart1.type = "col"
                chart1.style = 10
                chart1.title = "Top 15 Original Themes"
                chart1.x_axis.title = "Themes"
                chart1.y_axis.title = "Count"
                
                # Data range for original themes (assuming they start at column E)
                data = Reference(worksheet, min_col=6, min_row=1, max_row=min(16, len(top_original)+1), max_col=6)
                cats = Reference(worksheet, min_col=5, min_row=2, max_row=min(16, len(top_original)+1))
                
                chart1.add_data(data, titles_from_data=True)
                chart1.set_categories(cats)
                chart1.dataLabels = DataLabelList()
                chart1.dataLabels.showVal = True
                
                # Position chart
                worksheet.add_chart(chart1, "A25")
            
            # Consolidated themes chart
            if consolidated_counts:
                # Create chart
                chart2 = BarChart()
                chart2.type = "col"
                chart2.style = 11
                chart2.title = "Consolidated Themes Distribution"
                chart2.x_axis.title = "Themes"
                chart2.y_axis.title = "Count"
                
                # Data range for consolidated themes (assuming they start at column H)
                data = Reference(worksheet, min_col=9, min_row=1, max_row=len(consolidated_counts)+1, max_col=9)
                cats = Reference(worksheet, min_col=8, min_row=2, max_row=len(consolidated_counts)+1)
                
                chart2.add_data(data, titles_from_data=True)
                chart2.set_categories(cats)
                chart2.dataLabels = DataLabelList()
                chart2.dataLabels.showVal = True
                
                # Position chart
                worksheet.add_chart(chart2, "K25")
                
        except ImportError:
            logger.warning("openpyxl chart functionality not available - charts will not be created")
        except Exception as e:
            logger.warning(f"Failed to create charts: {str(e)}")
    
