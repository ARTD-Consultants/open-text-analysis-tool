"""Simple theme management with basic consolidation."""

import logging
from typing import Dict, List, Optional
from collections import Counter

from ..models.theme import SimpleThemeManager

logger = logging.getLogger(__name__)


class ThemeManager:
    """Simple theme management with basic consolidation."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize theme manager."""
        self.theme_manager = SimpleThemeManager()
        self.similarity_threshold = similarity_threshold
        
        # Statistics
        self.total_themes_processed = 0
        self.themes_merged = 0
    
    def add_theme_occurrence(self, theme_name: str, confidence: float = 1.0) -> str:
        """Add a theme occurrence."""
        if not theme_name or not theme_name.strip():
            return ""
        
        theme_name = theme_name.strip()
        self.total_themes_processed += 1
        
        # Simple case-insensitive deduplication
        existing_themes = self.theme_manager.get_all_themes()
        for existing_theme in existing_themes:
            if theme_name.lower() == existing_theme.lower():
                self.theme_manager.add_theme(existing_theme, confidence)
                return existing_theme
        
        # Add as new theme
        self.theme_manager.add_theme(theme_name, confidence)
        return theme_name
    
    def apply_theme_consolidation(self, results: List, ai_client=None, settings=None) -> None:
        """Apply basic theme consolidation across all results."""
        logger.info("Applying theme consolidation...")
        
        # Collect all themes with frequencies
        all_themes = []
        for result in results:
            for theme in result.original_themes:
                all_themes.append(theme)
        
        # Get theme frequencies
        theme_counts = Counter(all_themes)
        
        # Simple consolidation: merge very similar themes (exact match ignoring case)
        theme_mapping = {}
        processed_themes = set()
        
        for theme in theme_counts.keys():
            if theme in processed_themes:
                continue
                
            # Look for case-insensitive matches
            canonical_theme = theme
            for other_theme in theme_counts.keys():
                if (other_theme != theme and 
                    other_theme not in processed_themes and
                    theme.lower() == other_theme.lower()):
                    
                    # Merge into the more frequent one
                    if theme_counts[other_theme] > theme_counts[canonical_theme]:
                        canonical_theme = other_theme
                    
                    theme_mapping[other_theme] = canonical_theme
                    processed_themes.add(other_theme)
            
            processed_themes.add(canonical_theme)
        
        # Apply consolidation to results
        for result in results:
            consolidated_themes = []
            consolidated_confidences = {}
            
            for i, theme in enumerate(result.original_themes):
                final_theme = theme_mapping.get(theme, theme)
                
                if final_theme not in consolidated_themes:
                    consolidated_themes.append(final_theme)
                    confidence = result.original_theme_confidences.get(theme, 1.0)
                    consolidated_confidences[final_theme] = confidence
            
            # Update result with consolidated themes
            result.themes = consolidated_themes[:settings.max_themes_per_text if settings else 3]
            result.theme_confidences = consolidated_confidences
        
        logger.info(f"Theme consolidation completed. Merged {len(theme_mapping)} themes")
        self.themes_merged = len(theme_mapping)
    
    def get_theme_statistics(self) -> Dict:
        """Get theme management statistics."""
        return {
            "total_themes_processed": self.total_themes_processed,
            "unique_themes": self.theme_manager.get_total_themes(),
            "themes_merged": self.themes_merged,
            "total_occurrences": self.theme_manager.get_total_occurrences()
        }
    
    def get_sorted_themes(self) -> List:
        """Get themes sorted by frequency."""
        return self.theme_manager.get_sorted_themes()
    
    def get_theme_summary(self) -> Dict[str, int]:
        """Get theme summary as dict."""
        return self.theme_manager.get_theme_summary()