"""Theme management with consistency validation and hierarchy support."""

import logging
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter

from ..models.theme import Theme, ThemeHierarchy
from ..utils.similarity import SimilarityCalculator

logger = logging.getLogger(__name__)


class ThemeManager:
    """Manages themes with consistency validation and hierarchy support."""
    
    def __init__(
        self,
        similarity_calculator: Optional[SimilarityCalculator] = None,
        similarity_threshold: float = 0.85,
        enable_validation: bool = True,
        global_theme_limit: int = 200
    ):
        """Initialize theme manager."""
        self.hierarchy = ThemeHierarchy()
        self.similarity_calculator = similarity_calculator
        self.similarity_threshold = similarity_threshold
        self.enable_validation = enable_validation
        self.global_theme_limit = global_theme_limit
        
        # Track theme validation history
        self.validation_history: Dict[str, Dict[str, float]] = {}
        self.merge_history: List[Tuple[str, str, str]] = []  # (theme1, theme2, reason)
        
        # Statistics
        self.total_themes_processed = 0
        self.themes_merged = 0
        self.validation_calls = 0
        self.themes_rejected_for_limit = 0  # Themes converted to NA due to global limit
    
    def add_theme_occurrence(
        self,
        theme_name: str,
        confidence: float = 1.0,
        context: str = "",
        validate: bool = None
    ) -> str:
        """
        Add a theme occurrence, with optional validation against existing themes.
        
        Args:
            theme_name: Name of the theme
            confidence: Confidence score for this occurrence
            context: Context text for validation
            validate: Whether to validate similarity (uses instance default if None)
            
        Returns:
            Final theme name (may be different if merged with existing theme)
        """
        if validate is None:
            validate = self.enable_validation
        
        self.total_themes_processed += 1
        theme_name = theme_name.strip()
        
        if not theme_name:
            return ""
        
        # Global theme limit removed - allow unlimited themes for accurate analysis
        
        # Check if we should validate against existing themes
        if validate and self.similarity_calculator:
            canonical_theme = self._validate_and_merge_theme(theme_name, context)
            if canonical_theme != theme_name:
                logger.debug(f"Merged '{theme_name}' into '{canonical_theme}'")
                theme_name = canonical_theme
        
        # Find or create theme
        existing_theme = self.hierarchy.find_theme_by_name(theme_name)
        
        if existing_theme:
            existing_theme.add_occurrence(confidence)
        else:
            new_theme = Theme(name=theme_name)
            new_theme.add_occurrence(confidence)
            self.hierarchy.add_theme(new_theme)
        
        return theme_name
    
    def _validate_and_merge_theme(self, theme_name: str, context: str = "") -> str:
        """Validate theme against existing themes and potentially merge."""
        existing_themes = [theme.name for theme in self.hierarchy.get_all_themes_flat()]
        
        if not existing_themes:
            return theme_name
        
        # Check if we've already validated this theme
        if theme_name in self.validation_history:
            cached_results = self.validation_history[theme_name]
            # Find best match from cache
            best_match = max(cached_results.items(), key=lambda x: x[1])
            if best_match[1] >= self.similarity_threshold:
                return best_match[0]
            return theme_name
        
        # Find similar themes using embeddings
        similar_themes = self.similarity_calculator.find_similar_themes(
            theme_name, existing_themes, self.similarity_threshold
        )
        
        if similar_themes:
            # Take the most similar theme
            most_similar_theme, similarity_score = similar_themes[0]
            
            # Cache the validation result
            if theme_name not in self.validation_history:
                self.validation_history[theme_name] = {}
            self.validation_history[theme_name][most_similar_theme] = similarity_score
            
            self.validation_calls += 1
            logger.debug(f"Theme '{theme_name}' matched to '{most_similar_theme}' "
                        f"(similarity: {similarity_score:.3f})")
            
            # Record merge
            self.merge_history.append((theme_name, most_similar_theme, "similarity_based"))
            self.themes_merged += 1
            
            return most_similar_theme
        
        return theme_name
    
    def batch_add_themes(
        self,
        theme_data: List[Tuple[str, float]],
        context_texts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple themes efficiently.
        
        Args:
            theme_data: List of (theme_name, confidence) tuples
            context_texts: Optional context texts for validation
            
        Returns:
            List of final theme names
        """
        if context_texts is None:
            context_texts = [""] * len(theme_data)
        
        final_themes = []
        
        for i, (theme_name, confidence) in enumerate(theme_data):
            context = context_texts[i] if i < len(context_texts) else ""
            final_theme = self.add_theme_occurrence(theme_name, confidence, context)
            final_themes.append(final_theme)
        
        return final_themes
    
    def get_theme_suggestions(
        self,
        min_frequency: int = 2,
        max_suggestions: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Get suggestions for theme merges based on similarity."""
        if not self.similarity_calculator:
            return []
        
        themes = self.hierarchy.get_all_themes_flat()
        theme_frequencies = {theme.name: theme.frequency for theme in themes}
        
        # Filter themes by minimum frequency
        eligible_themes = {
            name: freq for name, freq in theme_frequencies.items() 
            if freq >= min_frequency
        }
        
        if len(eligible_themes) < 2:
            return []
        
        suggestions = self.similarity_calculator.suggest_theme_merges(
            eligible_themes, 
            self.similarity_threshold * 0.9,  # Slightly lower threshold for suggestions
            min_frequency_difference=5
        )
        
        return suggestions[:max_suggestions]
    
    def merge_themes(
        self,
        theme_name1: str,
        theme_name2: str,
        keep_first: bool = True,
        reason: str = "manual"
    ) -> bool:
        """
        Manually merge two themes.
        
        Args:
            theme_name1: First theme name
            theme_name2: Second theme name
            keep_first: Whether to keep the first theme name
            reason: Reason for merge
            
        Returns:
            Success status
        """
        theme1 = self.hierarchy.find_theme_by_name(theme_name1)
        theme2 = self.hierarchy.find_theme_by_name(theme_name2)
        
        if not theme1 or not theme2:
            logger.warning(f"Cannot merge themes - one or both not found: {theme_name1}, {theme_name2}")
            return False
        
        success = self.hierarchy.merge_themes(theme1.theme_id, theme2.theme_id, keep_first)
        
        if success:
            kept_name = theme_name1 if keep_first else theme_name2
            removed_name = theme_name2 if keep_first else theme_name1
            
            self.merge_history.append((removed_name, kept_name, reason))
            self.themes_merged += 1
            
            logger.info(f"Successfully merged '{removed_name}' into '{kept_name}' (reason: {reason})")
            
            # Update validation history
            if removed_name in self.validation_history:
                if kept_name not in self.validation_history:
                    self.validation_history[kept_name] = {}
                self.validation_history[kept_name].update(self.validation_history[removed_name])
                del self.validation_history[removed_name]
        
        return success
    
    def rename_theme(self, old_name: str, new_name: str) -> bool:
        """Rename a theme."""
        theme = self.hierarchy.find_theme_by_name(old_name)
        if not theme:
            return False
        
        # Check if new name already exists
        existing = self.hierarchy.find_theme_by_name(new_name)
        if existing:
            logger.warning(f"Cannot rename - theme '{new_name}' already exists")
            return False
        
        theme.name = new_name
        logger.info(f"Renamed theme '{old_name}' to '{new_name}'")
        return True
    
    def create_theme_hierarchy(
        self,
        parent_theme: str,
        child_themes: List[str]
    ) -> bool:
        """Create hierarchical relationships between themes."""
        parent = self.hierarchy.find_theme_by_name(parent_theme)
        
        if not parent:
            # Create parent theme if it doesn't exist
            parent = Theme(name=parent_theme)
            self.hierarchy.add_theme(parent)
        
        success_count = 0
        
        for child_name in child_themes:
            child = self.hierarchy.find_theme_by_name(child_name)
            
            if child:
                # Move child under parent
                if child.parent_id:
                    old_parent = self.hierarchy.get_parent(child.theme_id)
                    if old_parent:
                        old_parent.children_ids.discard(child.theme_id)
                
                child.parent_id = parent.theme_id
                parent.children_ids.add(child.theme_id)
                success_count += 1
        
        logger.info(f"Created hierarchy: '{parent_theme}' with {success_count} children")
        return success_count > 0
    
    def get_theme_statistics(self) -> Dict[str, any]:
        """Get comprehensive theme statistics."""
        themes = self.hierarchy.get_all_themes_flat()
        
        if not themes:
            return {
                "total_themes": 0,
                "total_occurrences": 0,
                "themes_processed": self.total_themes_processed,
                "themes_merged": self.themes_merged,
                "themes_rejected_for_limit": self.themes_rejected_for_limit,
                "global_theme_limit": self.global_theme_limit,
                "global_limit_reached": False,
                "validation_calls": self.validation_calls,
                "merge_rate": 0,
                "frequency_stats": {
                    "mean": 0,
                    "median": 0,
                    "min": 0,
                    "max": 0,
                    "total": 0
                },
                "confidence_stats": {
                    "mean": 0,
                    "min": 0,
                    "max": 0
                },
                "hierarchy_stats": self.hierarchy.get_statistics(),
                "top_themes": []
            }
        
        frequencies = [theme.frequency for theme in themes]
        confidences = [theme.average_confidence for theme in themes if theme.confidence_scores]
        
        return {
            "total_themes": len(themes),
            "total_occurrences": sum(frequencies),
            "themes_processed": self.total_themes_processed,
            "themes_merged": self.themes_merged,
            "themes_rejected_for_limit": self.themes_rejected_for_limit,
            "global_theme_limit": self.global_theme_limit,
            "global_limit_reached": len(self.hierarchy.get_all_themes_flat()) >= self.global_theme_limit,
            "validation_calls": self.validation_calls,
            "merge_rate": self.themes_merged / self.total_themes_processed if self.total_themes_processed > 0 else 0,
            "frequency_stats": {
                "mean": sum(frequencies) / len(frequencies),
                "median": sorted(frequencies)[len(frequencies) // 2],
                "min": min(frequencies),
                "max": max(frequencies),
                "total": sum(frequencies)
            },
            "confidence_stats": {
                "mean": sum(confidences) / len(confidences) if confidences else 0,
                "min": min(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 0
            },
            "hierarchy_stats": self.hierarchy.get_statistics(),
            "top_themes": [
                {"name": theme.name, "frequency": theme.frequency, "confidence": theme.average_confidence}
                for theme in themes[:10]
            ]
        }
    
    def export_themes(self) -> Dict[str, any]:
        """Export all themes with their relationships and statistics."""
        themes_data = {}
        
        for theme in self.hierarchy.get_all_themes_flat():
            theme_path = self.hierarchy.get_theme_path(theme.theme_id)
            parent = self.hierarchy.get_parent(theme.theme_id)
            children = self.hierarchy.get_children(theme.theme_id)
            
            themes_data[theme.name] = {
                "id": theme.theme_id,
                "frequency": theme.frequency,
                "average_confidence": theme.average_confidence,
                "confidence_scores": theme.confidence_scores,
                "hierarchy_path": theme_path,
                "parent": parent.name if parent else None,
                "children": [child.name for child in children],
                "level": len(theme_path) - 1
            }
        
        return {
            "themes": themes_data,
            "statistics": self.get_theme_statistics(),
            "merge_history": self.merge_history,
            "export_timestamp": logger.info.__self__.name if hasattr(logger.info, '__self__') else "unknown"
        }
    
    def get_themes_by_frequency(self, min_frequency: int = 1, max_results: int = None) -> List[Theme]:
        """Get themes filtered and sorted by frequency."""
        themes = self.hierarchy.get_all_themes_flat()
        filtered_themes = [theme for theme in themes if theme.frequency >= min_frequency]
        
        if max_results:
            filtered_themes = filtered_themes[:max_results]
        
        return filtered_themes
    
    def get_themes_by_confidence(self, min_confidence: float = 0.0, max_results: int = None) -> List[Theme]:
        """Get themes filtered and sorted by confidence."""
        themes = self.hierarchy.get_all_themes_flat()
        filtered_themes = [
            theme for theme in themes 
            if theme.average_confidence >= min_confidence
        ]
        
        # Sort by confidence descending
        filtered_themes.sort(key=lambda t: t.average_confidence, reverse=True)
        
        if max_results:
            filtered_themes = filtered_themes[:max_results]
        
        return filtered_themes
    
    def validate_theme_quality(self) -> Dict[str, any]:
        """Validate the quality of the current theme set."""
        themes = self.hierarchy.get_all_themes_flat()
        
        if not themes:
            return {"status": "empty", "issues": ["No themes found"]}
        
        issues = []
        warnings = []
        
        # Check for very low frequency themes
        low_freq_themes = [t for t in themes if t.frequency == 1]
        if len(low_freq_themes) > len(themes) * 0.5:
            issues.append(f"Many themes appear only once ({len(low_freq_themes)}/{len(themes)})")
        
        # Check for very short theme names
        short_names = [t for t in themes if len(t.name.split()) == 1]
        if len(short_names) > len(themes) * 0.8:
            warnings.append(f"Many themes have single-word names ({len(short_names)}/{len(themes)})")
        
        # Check for very long theme names
        long_names = [t for t in themes if len(t.name.split()) > 5]
        if long_names:
            warnings.append(f"Some themes have very long names ({len(long_names)} themes)")
        
        # Check theme distribution
        total_occurrences = sum(t.frequency for t in themes)
        top_10_occurrences = sum(t.frequency for t in themes[:10])
        
        if total_occurrences > 0 and top_10_occurrences / total_occurrences > 0.8:
            warnings.append("Theme distribution is heavily concentrated in top themes")
        
        # Determine overall status
        if issues:
            status = "poor"
        elif warnings:
            status = "fair"
        else:
            status = "good"
        
        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "theme_count": len(themes),
            "total_occurrences": total_occurrences,
            "single_occurrence_themes": len(low_freq_themes),
            "theme_distribution_score": 1 - (top_10_occurrences / total_occurrences) if total_occurrences > 0 else 0
        }

    def apply_theme_consolidation(
        self, 
        all_results: List, 
        openai_client, 
        settings
    ) -> Dict[str, str]:
        """
        Apply theme consolidation to analysis results using GPT-based mapping.
        
        Args:
            all_results: List of AnalysisResult objects
            openai_client: OpenAI client for GPT calls
            settings: Settings object with consolidation parameters
            
        Returns:
            Dictionary mapping original themes to consolidated themes
        """
        from ..config.prompts import Prompts
        
        logger.info(f"Applying theme consolidation to {len(all_results)} results")
        
        # Step 1: Collect all original themes with their frequencies and contexts
        theme_data = self._collect_theme_data(all_results)
        logger.info(f"Collected {len(theme_data)} unique original themes")
        
        # Step 2: Create representative themes using GPT
        original_theme_list = list(theme_data.keys())
        representative_themes = self._create_representative_themes_with_gpt(
            original_theme_list, theme_data, openai_client, settings
        )
        
        # Step 3: Map original themes to representative themes using chunked GPT calls
        final_theme_mapping = self._map_themes_with_chunked_gpt(
            original_theme_list, representative_themes, openai_client, settings
        )
        
        # Step 4: Apply consolidated themes to all results
        self._apply_mapping_to_results(all_results, final_theme_mapping, settings)
        
        # Step 5: Update theme hierarchy with consolidated theme frequencies
        self._update_theme_hierarchy_frequencies(all_results)
        
        # Calculate final statistics
        unique_final_themes = set(final_theme_mapping.values())
        unique_final_themes.discard("NA")  # Remove NA from count
        
        logger.info(f"Theme consolidation complete. Final unique themes: {len(unique_final_themes)}")
        
        return final_theme_mapping

    def _collect_theme_data(self, all_results: List) -> Dict[str, Dict]:
        """Collect theme data from analysis results."""
        theme_data = {}  # {theme_name: {'count': int, 'confidences': [float], 'contexts': [str]}}
        
        for result in all_results:
            for theme in result.original_themes:
                confidence = result.original_theme_confidences.get(theme, 1.0)
                
                if theme not in theme_data:
                    theme_data[theme] = {
                        'count': 0,
                        'confidences': [],
                        'contexts': []
                    }
                
                theme_data[theme]['count'] += 1
                theme_data[theme]['confidences'].append(confidence)
                theme_data[theme]['contexts'].append(result.original_text[:100])  # Sample context
        
        return theme_data

    def _create_representative_themes_with_gpt(
        self, 
        original_themes: List[str], 
        theme_data: Dict, 
        openai_client, 
        settings
    ) -> List[str]:
        """Create representative themes using GPT with configurable parameters."""
        from ..config.prompts import Prompts
        
        prompt = Prompts.representative_themes_creation_prompt(
            original_themes=original_themes,
            num_representative_themes=settings.representative_themes_count
        )

        try:
            response = openai_client.analyze_text_batch(
                prompt=prompt,
                temperature=settings.consolidation_temperature_representative,
                max_tokens=settings.consolidation_max_tokens_representative
            )
            
            # Parse response into list of themes
            representative_themes = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove numbering like "1." or "1:"
                    theme = line.split('.', 1)[-1].split(':', 1)[-1].strip()
                    if theme:
                        representative_themes.append(theme)
            
            # Ensure we have exactly the expected number of themes
            representative_themes = representative_themes[:settings.representative_themes_count]
            
            if len(representative_themes) < settings.representative_themes_count:
                logger.warning(f"Only got {len(representative_themes)} representative themes, expected {settings.representative_themes_count}")
            
            logger.info(f"Created {len(representative_themes)} representative themes: {representative_themes}")
            return representative_themes
            
        except Exception as e:
            logger.error(f"Failed to create representative themes: {e}")
            # Fallback: use most frequent themes or first N themes
            if theme_data:
                theme_counts = Counter({theme: data['count'] for theme, data in theme_data.items()})
                fallback_themes = [theme for theme, count in theme_counts.most_common(settings.representative_themes_count)]
            else:
                fallback_themes = original_themes[:settings.representative_themes_count]
            logger.info(f"Using fallback themes: {fallback_themes}")
            return fallback_themes

    def _map_themes_with_chunked_gpt(
        self, 
        original_themes: List[str], 
        representative_themes: List[str], 
        openai_client, 
        settings,
        chunk_size: int = None
    ) -> Dict[str, str]:
        """Map original themes to representative themes using chunked GPT calls."""
        from ..config.prompts import Prompts
        
        if chunk_size is None:
            chunk_size = settings.theme_mapping_chunk_size
            
        theme_mapping = {}
        
        # Process themes in chunks
        for i in range(0, len(original_themes), chunk_size):
            chunk = original_themes[i:i + chunk_size]
            
            prompt = Prompts.theme_mapping_prompt(
                original_themes_chunk=chunk,
                representative_themes=representative_themes
            )

            try:
                response = openai_client.analyze_text_batch(
                    prompt=prompt,
                    temperature=settings.consolidation_temperature_mapping,
                    max_tokens=settings.consolidation_max_tokens_mapping
                )
                
                # Parse mapping response
                chunk_mapping = self._parse_theme_mapping_response(response, chunk, representative_themes)
                theme_mapping.update(chunk_mapping)
                
                total_chunks = (len(original_themes) + chunk_size - 1) // chunk_size
                current_chunk = i // chunk_size + 1
                logger.info(f"Mapped chunk {current_chunk}/{total_chunks}: {len(chunk_mapping)} themes")
                
            except Exception as e:
                current_chunk = i // chunk_size + 1
                logger.error(f"Failed to map chunk {current_chunk}: {e}")
                # Fallback: map to first representative theme
                for theme in chunk:
                    theme_mapping[theme] = representative_themes[0] if representative_themes else "NA"
        
        logger.info(f"Theme mapping complete: {len(theme_mapping)} themes mapped")
        return theme_mapping

    def _parse_theme_mapping_response(
        self, 
        response: str, 
        original_chunk: List[str], 
        representative_themes: List[str]
    ) -> Dict[str, str]:
        """Parse GPT response for theme mappings."""
        mapping = {}
        
        # Parse line by line
        for line in response.strip().split('\n'):
            line = line.strip()
            if '->' in line:
                parts = line.split('->', 1)
                if len(parts) == 2:
                    original = parts[0].strip()
                    representative = parts[1].strip()
                    
                    # Validate that the representative theme exists
                    if representative in representative_themes:
                        mapping[original] = representative
                    else:
                        # Find closest match
                        best_match = self._find_closest_representative_theme(representative, representative_themes)
                        mapping[original] = best_match
                        logger.debug(f"Corrected mapping: '{original}' -> '{representative}' became '{best_match}'")
        
        # Handle any unmapped themes from the chunk
        for theme in original_chunk:
            if theme not in mapping:
                # Default to first representative theme
                mapping[theme] = representative_themes[0] if representative_themes else "NA"
                logger.warning(f"Theme '{theme}' not found in mapping, defaulting to '{mapping[theme]}'")
        
        return mapping

    def _find_closest_representative_theme(self, target: str, representative_themes: List[str]) -> str:
        """Find the closest representative theme by simple string matching."""
        target_lower = target.lower()
        
        # Look for exact substring matches first
        for theme in representative_themes:
            if target_lower in theme.lower() or theme.lower() in target_lower:
                return theme
        
        # Fallback to first theme
        return representative_themes[0] if representative_themes else "NA"

    def _apply_mapping_to_results(self, all_results: List, final_theme_mapping: Dict[str, str], settings):
        """Apply consolidated theme mapping to analysis results."""
        for result in all_results:
            consolidated_themes = []
            consolidated_confidences = {}
            
            for theme in result.original_themes:
                confidence = result.original_theme_confidences.get(theme, 1.0)
                consolidated_theme = final_theme_mapping.get(theme, "NA")
                
                if consolidated_theme not in consolidated_themes:
                    consolidated_themes.append(consolidated_theme)
                    consolidated_confidences[consolidated_theme] = confidence
            
            # Set consolidated themes
            result.themes = consolidated_themes
            result.theme_confidences = consolidated_confidences
            
            # Apply per-entry limits to consolidated themes
            result.apply_consolidation_limits(
                settings.max_themes_per_entry_consolidated,
                settings.theme_confidence_threshold
            )
    
    def _update_theme_hierarchy_frequencies(self, all_results: List):
        """Update theme hierarchy with frequencies from consolidated results."""
        # Clear existing themes first
        self.hierarchy = ThemeHierarchy()
        
        # Count consolidated theme frequencies
        theme_counts = Counter()
        theme_confidences = {}
        
        for result in all_results:
            for theme in result.themes:
                if theme and theme != "NA":
                    theme_counts[theme] += 1
                    confidence = result.theme_confidences.get(theme, 1.0)
                    if theme not in theme_confidences:
                        theme_confidences[theme] = []
                    theme_confidences[theme].append(confidence)
        
        # Create themes with proper frequencies
        for theme_name, frequency in theme_counts.items():
            theme = Theme(name=theme_name)
            # Set frequency and confidence scores
            theme.frequency = frequency
            if theme_name in theme_confidences:
                theme.confidence_scores = theme_confidences[theme_name]
            self.hierarchy.add_theme(theme)
        
        logger.info(f"Updated theme hierarchy with {len(theme_counts)} themes and their frequencies")