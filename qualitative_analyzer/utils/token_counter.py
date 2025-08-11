"""Token counting and estimation utilities."""

import tiktoken
from typing import List, Dict, Any


class TokenCounter:
    """Handles token counting for various models."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize with specified encoding."""
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to simple character-based estimation
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback estimation: roughly 4 characters per token
            return len(text) // 4
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for a batch of texts."""
        return [self.count_tokens(text) for text in texts]
    
    def estimate_prompt_tokens(
        self, 
        texts: List[str], 
        base_prompt: str,
        overhead_per_entry: int = 50
    ) -> int:
        """Estimate total tokens including prompt overhead."""
        text_tokens = sum(self.count_tokens(text) for text in texts)
        prompt_tokens = self.count_tokens(base_prompt)
        overhead_tokens = len(texts) * overhead_per_entry
        
        return text_tokens + prompt_tokens + overhead_tokens
    
    def get_batch_size_recommendation(
        self,
        texts: List[str],
        max_tokens: int = 4000,
        base_prompt_tokens: int = 500,
        overhead_per_entry: int = 50
    ) -> int:
        """Recommend optimal batch size based on token limits."""
        if not texts:
            return 0
        
        # Calculate average tokens per text
        sample_size = min(10, len(texts))
        sample_tokens = [self.count_tokens(text) for text in texts[:sample_size]]
        avg_tokens_per_text = sum(sample_tokens) / len(sample_tokens)
        
        # Account for base prompt and overhead
        available_tokens = max_tokens - base_prompt_tokens
        tokens_per_entry = avg_tokens_per_text + overhead_per_entry
        
        if tokens_per_entry <= 0:
            return 1
        
        recommended_size = int(available_tokens // tokens_per_entry)
        return max(1, min(recommended_size, len(texts)))
    
    def analyze_token_distribution(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze token distribution in a dataset."""
        token_counts = self.count_tokens_batch(texts)
        
        if not token_counts:
            return {"error": "No texts provided"}
        
        return {
            "total_texts": len(token_counts),
            "total_tokens": sum(token_counts),
            "mean_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "median_tokens": sorted(token_counts)[len(token_counts) // 2],
            "percentiles": {
                "25th": sorted(token_counts)[int(len(token_counts) * 0.25)],
                "75th": sorted(token_counts)[int(len(token_counts) * 0.75)],
                "90th": sorted(token_counts)[int(len(token_counts) * 0.90)],
                "95th": sorted(token_counts)[int(len(token_counts) * 0.95)]
            }
        }
    
    def create_optimal_batches(
        self,
        texts: List[str],
        max_tokens_per_batch: int = 4000,
        target_tokens_per_batch: int = 3000,
        base_prompt_tokens: int = 500,
        overhead_per_entry: int = 50
    ) -> List[List[int]]:
        """Create optimal batches by grouping text indices."""
        if not texts:
            return []
        
        # Calculate token counts for all texts
        token_counts = [(i, self.count_tokens(text)) 
                       for i, text in enumerate(texts)]
        
        # Sort by token count for better packing
        token_counts.sort(key=lambda x: x[1])
        
        batches = []
        current_batch = []
        current_tokens = base_prompt_tokens
        
        for text_idx, token_count in token_counts:
            entry_cost = token_count + overhead_per_entry
            
            # Check if adding this text would exceed limits
            if (current_tokens + entry_cost > max_tokens_per_batch and 
                current_batch):
                # Save current batch and start new one
                batches.append(current_batch)
                current_batch = []
                current_tokens = base_prompt_tokens
            
            # Add text to current batch
            current_batch.append(text_idx)
            current_tokens += entry_cost
            
            # If batch reaches target size, save it
            if current_tokens >= target_tokens_per_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = base_prompt_tokens
        
        # Add final batch if it has content
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def estimate_processing_cost(
        self,
        texts: List[str],
        cost_per_1k_tokens: float = 0.01,
        base_prompt_tokens: int = 500,
        overhead_per_entry: int = 50
    ) -> Dict[str, float]:
        """Estimate processing cost for a dataset."""
        total_text_tokens = sum(self.count_tokens(text) for text in texts)
        
        # Estimate number of API calls needed
        avg_tokens_per_text = total_text_tokens / len(texts) if texts else 0
        tokens_per_entry = avg_tokens_per_text + overhead_per_entry
        estimated_batch_size = max(1, int(4000 / tokens_per_entry))
        estimated_batches = (len(texts) + estimated_batch_size - 1) // estimated_batch_size
        
        # Calculate total tokens including prompts and overhead
        prompt_tokens = estimated_batches * base_prompt_tokens
        overhead_tokens = len(texts) * overhead_per_entry
        total_tokens = total_text_tokens + prompt_tokens + overhead_tokens
        
        # Calculate costs
        cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            "total_tokens": total_tokens,
            "text_tokens": total_text_tokens,
            "prompt_tokens": prompt_tokens,
            "overhead_tokens": overhead_tokens,
            "estimated_batches": estimated_batches,
            "estimated_cost_usd": round(cost, 4),
            "cost_per_text": round(cost / len(texts), 6) if texts else 0
        }