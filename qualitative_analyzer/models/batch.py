"""Batch processing models and configuration."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    
    max_batch_size: int = 30
    max_tokens_per_batch: int = 4000
    target_tokens_per_batch: int = 3000
    token_overhead_per_entry: int = 200  # Estimated prompt overhead
    min_batch_size: int = 1
    enable_dynamic_sizing: bool = True
    
    def get_optimal_batch_size(self, average_text_length: int) -> int:
        """Calculate optimal batch size based on text length."""
        if not self.enable_dynamic_sizing:
            return self.max_batch_size
        
        # Estimate tokens (rough approximation: 1 token = 4 characters)
        estimated_tokens_per_text = average_text_length // 4
        
        # Calculate how many texts can fit considering overhead
        total_overhead = self.token_overhead_per_entry
        available_tokens = self.target_tokens_per_batch - total_overhead
        
        if available_tokens <= 0:
            return self.min_batch_size
        
        optimal_size = available_tokens // (estimated_tokens_per_text + self.token_overhead_per_entry)
        
        # Ensure within bounds
        return max(self.min_batch_size, min(optimal_size, self.max_batch_size))


@dataclass
class Batch:
    """Represents a batch of texts to be processed."""
    
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    texts: List[str] = field(default_factory=list)
    text_indices: List[int] = field(default_factory=list)  # Original indices in dataset
    estimated_tokens: int = 0
    actual_tokens: Optional[int] = None
    processing_status: str = "pending"  # pending, processing, completed, failed
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_text(self, text: str, original_index: int, estimated_tokens: int = 0) -> None:
        """Add a text to the batch."""
        self.texts.append(text)
        self.text_indices.append(original_index)
        self.estimated_tokens += estimated_tokens
    
    def can_add_text(self, estimated_tokens: int, config: BatchConfig) -> bool:
        """Check if a text can be added to this batch."""
        if len(self.texts) >= config.max_batch_size:
            return False
        
        total_estimated = self.estimated_tokens + estimated_tokens + config.token_overhead_per_entry
        return total_estimated <= config.max_tokens_per_batch
    
    def is_ready_for_processing(self) -> bool:
        """Check if batch is ready for processing."""
        return (self.processing_status == "pending" and 
                len(self.texts) > 0 and 
                self.retry_count < self.max_retries)
    
    def mark_processing(self) -> None:
        """Mark batch as currently processing."""
        self.processing_status = "processing"
    
    def mark_completed(self, actual_tokens: int = 0) -> None:
        """Mark batch as completed successfully."""
        self.processing_status = "completed"
        self.actual_tokens = actual_tokens
    
    def mark_failed(self, error_message: str) -> None:
        """Mark batch as failed."""
        self.processing_status = "failed"
        self.error_message = error_message
        self.retry_count += 1
    
    def can_retry(self) -> bool:
        """Check if batch can be retried."""
        return (self.processing_status == "failed" and 
                self.retry_count < self.max_retries)
    
    def reset_for_retry(self) -> None:
        """Reset batch status for retry."""
        if self.can_retry():
            self.processing_status = "pending"
            self.error_message = None
    
    def get_token_efficiency(self) -> Optional[float]:
        """Calculate token efficiency (actual vs estimated)."""
        if self.actual_tokens is None or self.estimated_tokens == 0:
            return None
        return self.actual_tokens / self.estimated_tokens
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary information about the batch."""
        return {
            "batch_id": self.batch_id,
            "text_count": len(self.texts),
            "estimated_tokens": self.estimated_tokens,
            "actual_tokens": self.actual_tokens,
            "token_efficiency": self.get_token_efficiency(),
            "processing_status": self.processing_status,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class BatchManager:
    """Manages the creation and tracking of batches."""
    
    config: BatchConfig = field(default_factory=BatchConfig)
    batches: List[Batch] = field(default_factory=list)
    completed_batches: int = 0
    failed_batches: int = 0
    
    def create_batches_from_texts(
        self, 
        texts: List[str], 
        token_estimator: Optional[callable] = None
    ) -> List[Batch]:
        """Create optimal batches from a list of texts."""
        if not texts:
            return []
        
        # Simple token estimation if no estimator provided
        if token_estimator is None:
            token_estimator = lambda text: len(text) // 4  # Rough approximation
        
        batches = []
        current_batch = Batch()
        
        # Sort texts by length for better packing
        text_items = [(i, text, token_estimator(text)) 
                     for i, text in enumerate(texts)]
        text_items.sort(key=lambda x: x[2])  # Sort by token count
        
        for original_index, text, estimated_tokens in text_items:
            # Check if current batch can accommodate this text
            if not current_batch.can_add_text(estimated_tokens, self.config):
                # Start new batch if current one has texts
                if current_batch.texts:
                    batches.append(current_batch)
                    current_batch = Batch()
                
                # If single text is too large, create a batch anyway (with warning)
                if not current_batch.texts:
                    current_batch.metadata["oversized"] = True
            
            current_batch.add_text(text, original_index, estimated_tokens)
        
        # Add the final batch if it has texts
        if current_batch.texts:
            batches.append(current_batch)
        
        # Store batches and return
        self.batches.extend(batches)
        return batches
    
    def get_pending_batches(self) -> List[Batch]:
        """Get all batches ready for processing."""
        return [batch for batch in self.batches if batch.is_ready_for_processing()]
    
    def get_failed_batches(self) -> List[Batch]:
        """Get all failed batches that can be retried."""
        return [batch for batch in self.batches if batch.can_retry()]
    
    def update_batch_status(self, batch_id: str, status: str, **kwargs) -> bool:
        """Update the status of a specific batch."""
        for batch in self.batches:
            if batch.batch_id == batch_id:
                if status == "processing":
                    batch.mark_processing()
                elif status == "completed":
                    batch.mark_completed(kwargs.get("actual_tokens", 0))
                    self.completed_batches += 1
                elif status == "failed":
                    batch.mark_failed(kwargs.get("error_message", "Unknown error"))
                    if batch.retry_count >= batch.max_retries:
                        self.failed_batches += 1
                return True
        return False
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about batch processing."""
        total_batches = len(self.batches)
        pending = len([b for b in self.batches if b.processing_status == "pending"])
        processing = len([b for b in self.batches if b.processing_status == "processing"])
        completed = self.completed_batches
        failed = self.failed_batches
        
        return {
            "total_batches": total_batches,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "completion_rate": completed / total_batches if total_batches > 0 else 0,
            "failure_rate": failed / total_batches if total_batches > 0 else 0
        }