"""Semantic similarity calculations using embeddings."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib
import os


class SimilarityCalculator:
    """Handles semantic similarity calculations for theme management."""
    
    def __init__(self, embedding_client=None, cache_dir: str = "embeddings_cache"):
        """Initialize with optional embedding client and cache directory."""
        self.embedding_client = embedding_client
        self.cache_dir = cache_dir
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Create cache directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_cache()
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self) -> None:
        """Load embeddings from cache file."""
        cache_file = os.path.join(self.cache_dir, "embeddings_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
            except Exception:
                self.embeddings_cache = {}
    
    def _save_cache(self) -> None:
        """Save embeddings to cache file."""
        if not self.cache_dir:
            return
        
        cache_file = os.path.join(self.cache_dir, "embeddings_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception:
            pass  # Fail silently if can't cache
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text, using cache if available."""
        text_hash = self._get_text_hash(text)
        
        # Check cache first
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        # Get embedding from client
        if self.embedding_client:
            try:
                embedding = self.embedding_client.get_embedding(text)
                if embedding is not None:
                    embedding_array = np.array(embedding)
                    self.embeddings_cache[text_hash] = embedding_array
                    self._save_cache()
                    return embedding_array
            except Exception:
                pass
        
        return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts efficiently."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[text_hash])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Get embeddings for uncached texts
        if uncached_texts and self.embedding_client:
            try:
                batch_embeddings = self.embedding_client.get_embeddings_batch(uncached_texts)
                
                for idx, embedding in zip(uncached_indices, batch_embeddings):
                    if embedding is not None:
                        embedding_array = np.array(embedding)
                        embeddings[idx] = embedding_array
                        
                        # Cache the embedding
                        text_hash = self._get_text_hash(texts[idx])
                        self.embeddings_cache[text_hash] = embedding_array
                
                self._save_cache()
                
            except Exception:
                pass  # Continue with None values for failed embeddings
        
        return embeddings
    
    def calculate_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> Optional[float]:
        """Calculate cosine similarity between two texts."""
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        if embedding1 is None or embedding2 is None:
            return None
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            embedding1.reshape(1, -1), 
            embedding2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def find_similar_themes(
        self,
        theme_name: str,
        existing_themes: List[str],
        threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """Find existing themes similar to a new theme."""
        if not existing_themes:
            return []
        
        similarities = []
        theme_embedding = self.get_embedding(theme_name)
        
        if theme_embedding is None:
            return []
        
        # Get embeddings for existing themes
        existing_embeddings = self.get_embeddings_batch(existing_themes)
        
        for theme, embedding in zip(existing_themes, existing_embeddings):
            if embedding is not None:
                similarity = cosine_similarity(
                    theme_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                
                if similarity >= threshold:
                    similarities.append((theme, float(similarity)))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def cluster_themes(
        self,
        themes: List[str],
        similarity_threshold: float = 0.8
    ) -> List[List[str]]:
        """Cluster similar themes together."""
        if len(themes) <= 1:
            return [themes] if themes else []
        
        # Get embeddings for all themes
        embeddings = self.get_embeddings_batch(themes)
        
        # Filter out themes without embeddings
        valid_themes = []
        valid_embeddings = []
        
        for theme, embedding in zip(themes, embeddings):
            if embedding is not None:
                valid_themes.append(theme)
                valid_embeddings.append(embedding)
        
        if len(valid_embeddings) <= 1:
            return [[theme] for theme in valid_themes]
        
        # Calculate similarity matrix
        embedding_matrix = np.array(valid_embeddings)
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Simple clustering based on threshold
        clusters = []
        assigned = set()
        
        for i, theme in enumerate(valid_themes):
            if theme in assigned:
                continue
            
            cluster = [theme]
            assigned.add(theme)
            
            # Find similar themes
            for j, other_theme in enumerate(valid_themes):
                if (i != j and 
                    other_theme not in assigned and 
                    similarity_matrix[i][j] >= similarity_threshold):
                    cluster.append(other_theme)
                    assigned.add(other_theme)
            
            clusters.append(cluster)
        
        return clusters
    
    def get_theme_similarity_matrix(
        self,
        themes: List[str]
    ) -> Optional[np.ndarray]:
        """Get full similarity matrix for themes."""
        embeddings = self.get_embeddings_batch(themes)
        
        # Filter valid embeddings
        valid_embeddings = [e for e in embeddings if e is not None]
        
        if len(valid_embeddings) < 2:
            return None
        
        embedding_matrix = np.array(valid_embeddings)
        return cosine_similarity(embedding_matrix)
    
    def suggest_theme_merges(
        self,
        theme_frequencies: Dict[str, int],
        similarity_threshold: float = 0.85,
        min_frequency_difference: int = 2
    ) -> List[Tuple[str, str, float]]:
        """Suggest theme merges based on similarity and frequency."""
        themes = list(theme_frequencies.keys())
        suggestions = []
        
        if len(themes) < 2:
            return suggestions
        
        # Get similarity matrix
        similarity_matrix = self.get_theme_similarity_matrix(themes)
        
        if similarity_matrix is None:
            return suggestions
        
        # Find similar theme pairs
        for i in range(len(themes)):
            for j in range(i + 1, len(themes)):
                similarity = similarity_matrix[i][j]
                
                if similarity >= similarity_threshold:
                    theme1, theme2 = themes[i], themes[j]
                    freq1, freq2 = theme_frequencies[theme1], theme_frequencies[theme2]
                    
                    # Only suggest merge if frequency difference isn't too large
                    freq_ratio = max(freq1, freq2) / max(min(freq1, freq2), 1)
                    
                    if freq_ratio <= min_frequency_difference:
                        suggestions.append((theme1, theme2, similarity))
        
        # Sort by similarity descending
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions
    
    def validate_theme_similarity(
        self,
        new_theme: str,
        existing_theme: str,
        context_text: str = ""
    ) -> Dict[str, Any]:
        """Validate if two themes represent the same concept."""
        # Basic similarity
        similarity = self.calculate_similarity(new_theme, existing_theme)
        
        result = {
            "similarity_score": similarity,
            "are_similar": similarity is not None and similarity > 0.8,
            "confidence": "high" if similarity and similarity > 0.9 else 
                         "medium" if similarity and similarity > 0.8 else "low"
        }
        
        # Enhanced validation with context if available
        if context_text and similarity:
            context_sim_new = self.calculate_similarity(new_theme, context_text)
            context_sim_existing = self.calculate_similarity(existing_theme, context_text)
            
            if context_sim_new and context_sim_existing:
                context_difference = abs(context_sim_new - context_sim_existing)
                result["context_similarity_difference"] = context_difference
                result["context_supports_merge"] = context_difference < 0.1
        
        return result