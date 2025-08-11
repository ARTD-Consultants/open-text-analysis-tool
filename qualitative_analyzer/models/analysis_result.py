"""Analysis result models for storing and managing analysis outcomes."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class AnalysisResult:
    """Represents the analysis result for a single text entry."""
    
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_text: str = ""
    summary: str = ""
    themes: List[str] = field(default_factory=list)  # Final consolidated themes
    theme_confidences: Dict[str, float] = field(default_factory=dict)
    original_themes: List[str] = field(default_factory=list)  # Raw AI-extracted themes
    original_theme_confidences: Dict[str, float] = field(default_factory=dict)  # Original confidences
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    emotion_score: Optional[float] = None
    emotion_label: Optional[str] = None
    processing_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_original_theme(self, theme_name: str, confidence: float = 1.0) -> None:
        """Add an original AI-extracted theme."""
        if theme_name not in self.original_themes:
            self.original_themes.append(theme_name)
        self.original_theme_confidences[theme_name] = confidence
    
    def add_theme(self, theme_name: str, confidence: float = 1.0) -> None:
        """Add a consolidated theme with confidence score."""
        if theme_name not in self.themes:
            self.themes.append(theme_name)
        self.theme_confidences[theme_name] = confidence
    
    def apply_consolidation_limits(self, max_themes_per_entry: int, min_confidence: float = 0.6) -> None:
        """Apply per-entry limits to consolidated themes."""
        if not self.themes:
            return
        
        # Sort themes by confidence (descending)
        theme_confidence_pairs = [(theme, self.theme_confidences.get(theme, 1.0)) for theme in self.themes]
        theme_confidence_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top N themes with sufficient confidence
        kept_themes = []
        kept_confidences = {}
        
        for theme, confidence in theme_confidence_pairs:
            if len(kept_themes) < max_themes_per_entry and confidence >= min_confidence:
                kept_themes.append(theme)
                kept_confidences[theme] = confidence
            else:
                # Add NA for filtered themes
                if "NA" not in kept_themes and len(kept_themes) < max_themes_per_entry:
                    kept_themes.append("NA")
                    kept_confidences["NA"] = 0.0
        
        # Update themes and confidences
        self.themes = kept_themes
        self.theme_confidences = kept_confidences
    
    def get_high_confidence_themes(self, threshold: float = 0.8) -> List[str]:
        """Get themes with confidence above threshold."""
        return [theme for theme in self.themes 
                if self.theme_confidences.get(theme, 1.0) >= threshold]
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence across all themes."""
        if not self.theme_confidences:
            return 1.0
        return sum(self.theme_confidences.values()) / len(self.theme_confidences)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "original_text": self.original_text,
            "summary": self.summary,
            "themes": self.themes,
            "theme_confidences": self.theme_confidences,
            "original_themes": self.original_themes,
            "original_theme_confidences": self.original_theme_confidences,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "emotion_score": self.emotion_score,
            "emotion_label": self.emotion_label,
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class BatchResult:
    """Represents the results from processing a batch of texts."""
    
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    results: List[AnalysisResult] = field(default_factory=list)
    processing_start: datetime = field(default_factory=datetime.now)
    processing_end: Optional[datetime] = None
    tokens_used: int = 0
    api_calls: int = 0
    errors: List[str] = field(default_factory=list)
    batch_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: AnalysisResult) -> None:
        """Add an analysis result to this batch."""
        self.results.append(result)
    
    def mark_completed(self) -> None:
        """Mark the batch as completed."""
        self.processing_end = datetime.now()
    
    def get_processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if not self.processing_end:
            return None
        return (self.processing_end - self.processing_start).total_seconds()
    
    def get_themes_summary(self) -> Dict[str, int]:
        """Get a summary of theme frequencies in this batch."""
        theme_counts = {}
        for result in self.results:
            for theme in result.themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        return theme_counts
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all results in batch."""
        if not self.results:
            return 0.0
        
        total_confidence = sum(result.get_average_confidence() for result in self.results)
        return total_confidence / len(self.results)
    
    def get_success_rate(self) -> float:
        """Calculate the success rate (non-error results / total)."""
        if not self.results:
            return 0.0
        
        successful = len([r for r in self.results if r.themes])
        return successful / len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "results": [result.to_dict() for result in self.results],
            "processing_start": self.processing_start.isoformat(),
            "processing_end": self.processing_end.isoformat() if self.processing_end else None,
            "processing_duration": self.get_processing_duration(),
            "tokens_used": self.tokens_used,
            "api_calls": self.api_calls,
            "errors": self.errors,
            "themes_summary": self.get_themes_summary(),
            "average_confidence": self.get_average_confidence(),
            "success_rate": self.get_success_rate(),
            "batch_metadata": self.batch_metadata
        }


@dataclass
class AnalysisSession:
    """Represents a complete analysis session with multiple batches."""
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_start: datetime = field(default_factory=datetime.now)
    session_end: Optional[datetime] = None
    batches: List[BatchResult] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    total_entries: int = 0
    total_tokens_used: int = 0
    total_api_calls: int = 0
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_batch(self, batch: BatchResult) -> None:
        """Add a batch result to this session."""
        self.batches.append(batch)
        self.total_tokens_used += batch.tokens_used
        self.total_api_calls += batch.api_calls
    
    def mark_completed(self) -> None:
        """Mark the session as completed."""
        self.session_end = datetime.now()
    
    def get_all_results(self) -> List[AnalysisResult]:
        """Get all analysis results from all batches."""
        results = []
        for batch in self.batches:
            results.extend(batch.results)
        return results
    
    def get_session_duration(self) -> Optional[float]:
        """Get total session duration in seconds."""
        if not self.session_end:
            return None
        return (self.session_end - self.session_start).total_seconds()
    
    def get_global_theme_summary(self) -> Dict[str, int]:
        """Get theme frequencies across all batches."""
        theme_counts = {}
        for batch in self.batches:
            batch_themes = batch.get_themes_summary()
            for theme, count in batch_themes.items():
                theme_counts[theme] = theme_counts.get(theme, 0) + count
        return theme_counts
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        all_results = self.get_all_results()
        
        return {
            "session_id": self.session_id,
            "total_batches": len(self.batches),
            "total_entries": len(all_results),
            "total_tokens_used": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "session_duration": self.get_session_duration(),
            "average_batch_size": len(all_results) / len(self.batches) if self.batches else 0,
            "tokens_per_entry": self.total_tokens_used / len(all_results) if all_results else 0,
            "unique_themes": len(self.get_global_theme_summary()),
            "average_themes_per_entry": sum(len(r.themes) for r in all_results) / len(all_results) if all_results else 0,
            "overall_confidence": sum(r.get_average_confidence() for r in all_results) / len(all_results) if all_results else 0
        }