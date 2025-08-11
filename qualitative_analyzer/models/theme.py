"""Theme data models and hierarchical structures."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import uuid


@dataclass
class Theme:
    """Represents a single theme with metadata."""
    
    name: str
    frequency: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    theme_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score for this theme."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    def add_occurrence(self, confidence: float = 1.0) -> None:
        """Add an occurrence of this theme with optional confidence score."""
        self.frequency += 1
        self.confidence_scores.append(confidence)
    
    def merge_with(self, other: "Theme") -> None:
        """Merge another theme into this one."""
        self.frequency += other.frequency
        self.confidence_scores.extend(other.confidence_scores)
        
        # Keep the more common name
        if other.frequency > self.frequency - other.frequency:
            self.name = other.name


@dataclass
class ThemeHierarchy:
    """Manages hierarchical relationships between themes."""
    
    themes: Dict[str, Theme] = field(default_factory=dict)
    root_themes: Set[str] = field(default_factory=set)
    
    def add_theme(self, theme: Theme, parent_id: Optional[str] = None) -> None:
        """Add a theme to the hierarchy."""
        self.themes[theme.theme_id] = theme
        
        if parent_id:
            theme.parent_id = parent_id
            if parent_id in self.themes:
                self.themes[parent_id].children_ids.add(theme.theme_id)
        else:
            self.root_themes.add(theme.theme_id)
    
    def get_theme(self, theme_id: str) -> Optional[Theme]:
        """Get a theme by ID."""
        return self.themes.get(theme_id)
    
    def find_theme_by_name(self, name: str) -> Optional[Theme]:
        """Find a theme by name."""
        for theme in self.themes.values():
            if theme.name.lower() == name.lower():
                return theme
        return None
    
    def get_children(self, theme_id: str) -> List[Theme]:
        """Get all child themes of a given theme."""
        theme = self.themes.get(theme_id)
        if not theme:
            return []
        
        return [self.themes[child_id] for child_id in theme.children_ids 
                if child_id in self.themes]
    
    def get_parent(self, theme_id: str) -> Optional[Theme]:
        """Get the parent theme of a given theme."""
        theme = self.themes.get(theme_id)
        if not theme or not theme.parent_id:
            return None
        
        return self.themes.get(theme.parent_id)
    
    def get_root_themes(self) -> List[Theme]:
        """Get all root-level themes."""
        return [self.themes[theme_id] for theme_id in self.root_themes 
                if theme_id in self.themes]
    
    def get_theme_path(self, theme_id: str) -> List[str]:
        """Get the full hierarchical path for a theme."""
        path = []
        current_theme = self.themes.get(theme_id)
        
        while current_theme:
            path.append(current_theme.name)
            if current_theme.parent_id:
                current_theme = self.themes.get(current_theme.parent_id)
            else:
                break
        
        return list(reversed(path))
    
    def get_all_themes_flat(self) -> List[Theme]:
        """Get all themes as a flat list, sorted by frequency."""
        return sorted(self.themes.values(), key=lambda t: t.frequency, reverse=True)
    
    def merge_themes(self, theme_id1: str, theme_id2: str, keep_first: bool = True) -> bool:
        """Merge two themes together."""
        theme1 = self.themes.get(theme_id1)
        theme2 = self.themes.get(theme_id2)
        
        if not theme1 or not theme2:
            return False
        
        if keep_first:
            theme1.merge_with(theme2)
            # Update parent-child relationships
            for child_id in theme2.children_ids:
                if child_id in self.themes:
                    self.themes[child_id].parent_id = theme_id1
                    theme1.children_ids.add(child_id)
            
            # Remove theme2 from hierarchy
            self._remove_theme(theme_id2)
        else:
            theme2.merge_with(theme1)
            # Update parent-child relationships
            for child_id in theme1.children_ids:
                if child_id in self.themes:
                    self.themes[child_id].parent_id = theme_id2
                    theme2.children_ids.add(child_id)
            
            # Remove theme1 from hierarchy
            self._remove_theme(theme_id1)
        
        return True
    
    def _remove_theme(self, theme_id: str) -> None:
        """Remove a theme from the hierarchy."""
        theme = self.themes.get(theme_id)
        if not theme:
            return
        
        # Remove from parent's children
        if theme.parent_id and theme.parent_id in self.themes:
            self.themes[theme.parent_id].children_ids.discard(theme_id)
        
        # Remove from root themes if applicable
        self.root_themes.discard(theme_id)
        
        # Remove from themes dict
        del self.themes[theme_id]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the theme hierarchy."""
        total_themes = len(self.themes)
        root_themes = len(self.root_themes)
        themes_with_children = sum(1 for t in self.themes.values() if t.children_ids)
        
        return {
            "total_themes": total_themes,
            "root_themes": root_themes,
            "themes_with_children": themes_with_children,
            "leaf_themes": total_themes - themes_with_children
        }