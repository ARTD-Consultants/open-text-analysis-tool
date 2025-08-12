"""Simple theme counting models."""

from typing import Dict, List, Tuple
from collections import defaultdict


class SimpleThemeManager:
    """Simple theme counting and management."""
    
    def __init__(self):
        """Initialize simple theme manager."""
        self.theme_counts = defaultdict(int)
        self.theme_confidences = defaultdict(list)
    
    def add_theme(self, name: str, confidence: float = 1.0) -> None:
        """Add a theme occurrence with confidence score."""
        if not name or not name.strip():
            return
            
        name = name.strip()
        self.theme_counts[name] += 1
        self.theme_confidences[name].append(confidence)
    
    def get_sorted_themes(self) -> List[Tuple[str, int]]:
        """Get themes sorted by frequency (descending)."""
        return sorted(self.theme_counts.items(), key=lambda x: x[1], reverse=True)
    
    def get_theme_count(self, name: str) -> int:
        """Get count for a specific theme."""
        return self.theme_counts.get(name, 0)
    
    def get_theme_confidence(self, name: str) -> float:
        """Get average confidence for a theme."""
        confidences = self.theme_confidences.get(name, [])
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def get_all_themes(self) -> List[str]:
        """Get all theme names."""
        return list(self.theme_counts.keys())
    
    def get_theme_summary(self) -> Dict[str, int]:
        """Get theme summary as dict."""
        return dict(self.theme_counts)
    
    def merge_theme(self, from_theme: str, to_theme: str) -> bool:
        """Merge one theme into another."""
        if from_theme not in self.theme_counts or to_theme not in self.theme_counts:
            return False
        
        # Merge counts and confidences
        self.theme_counts[to_theme] += self.theme_counts[from_theme]
        self.theme_confidences[to_theme].extend(self.theme_confidences[from_theme])
        
        # Remove old theme
        del self.theme_counts[from_theme]
        del self.theme_confidences[from_theme]
        
        return True
    
    def get_total_themes(self) -> int:
        """Get total number of unique themes."""
        return len(self.theme_counts)
    
    def get_total_occurrences(self) -> int:
        """Get total theme occurrences across all themes."""
        return sum(self.theme_counts.values())