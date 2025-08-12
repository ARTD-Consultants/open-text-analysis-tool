"""Simple batch processing for text analysis."""

import logging
from typing import List, Dict, Any, Optional, Callable
import time

# Simplified batch processor - no longer needs complex batch models
from ..models.analysis_result import BatchResult
from ..utils.token_counter import TokenCounter
from ..config.prompts import Prompts

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Simple batch processing for text analysis."""
    
    def __init__(
        self,
        token_counter: TokenCounter,
        batch_size: int = 15,
        max_tokens: int = 4000
    ):
        """Initialize batch processor."""
        self.token_counter = token_counter
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        
        # Processing statistics
        self.total_batches_processed = 0
        self.total_texts_processed = 0
        self.total_processing_time = 0.0
        self.failed_batches = 0
    
    def create_batches(self, texts: List[str]) -> List[List[str]]:
        """Create simple batches of texts."""
        if not texts:
            return []
        
        logger.info(f"Creating batches for {len(texts)} texts (batch_size={self.batch_size})")
        
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches")
        return batches
    
    def process_batches(
        self,
        batches: List[List[str]],
        processor_function: Callable[[List[str]], List[Dict[str, Any]]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """Process batches sequentially."""
        all_results = []
        start_time = time.time()
        
        for i, batch in enumerate(batches):
            try:
                results = processor_function(batch)
                all_results.extend(results)
                
                # Update statistics
                self.total_batches_processed += 1
                self.total_texts_processed += len(batch)
                
                if progress_callback:
                    progress_callback(i + 1, len(batches))
                
                logger.debug(f"Processed batch {i+1}/{len(batches)} ({len(batch)} texts)")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i+1}: {str(e)}")
                self.failed_batches += 1
                # Create empty results for failed batch
                all_results.extend([{"error": str(e)} for _ in batch])
        
        self.total_processing_time += time.time() - start_time
        return all_results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        avg_processing_time = (
            self.total_processing_time / self.total_batches_processed
            if self.total_batches_processed > 0 else 0
        )
        
        success_rate = (
            (self.total_batches_processed - self.failed_batches) / self.total_batches_processed
            if self.total_batches_processed > 0 else 0
        )
        
        return {
            "total_batches_processed": self.total_batches_processed,
            "total_texts_processed": self.total_texts_processed,
            "failed_batches": self.failed_batches,
            "success_rate": success_rate,
            "total_processing_time": self.total_processing_time,
            "average_processing_time_per_batch": avg_processing_time,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens
        }