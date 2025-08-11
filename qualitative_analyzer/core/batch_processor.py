"""Intelligent batch processing with dynamic sizing and parallel support."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..models.batch import Batch, BatchConfig, BatchManager
from ..models.analysis_result import BatchResult, AnalysisResult
from ..utils.token_counter import TokenCounter
from ..config.prompts import Prompts

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles intelligent batch processing with dynamic sizing."""
    
    def __init__(
        self,
        token_counter: TokenCounter,
        batch_config: Optional[BatchConfig] = None,
        max_concurrent_batches: int = 3
    ):
        """Initialize batch processor."""
        self.token_counter = token_counter
        self.batch_config = batch_config or BatchConfig()
        self.max_concurrent_batches = max_concurrent_batches
        
        # Processing statistics
        self.total_batches_processed = 0
        self.total_texts_processed = 0
        self.total_processing_time = 0.0
        self.failed_batches = 0
    
    def create_optimized_batches(
        self,
        texts: List[str],
        existing_themes: Optional[List[str]] = None
    ) -> List[Batch]:
        """Create optimized batches based on text length and token usage."""
        if not texts:
            return []
        
        logger.info(f"Creating optimized batches for {len(texts)} texts")
        
        # Analyze token distribution
        token_analysis = self.token_counter.analyze_token_distribution(texts)
        logger.info(f"Token analysis: mean={token_analysis['mean_tokens']:.1f}, "
                   f"max={token_analysis['max_tokens']}, "
                   f"95th percentile={token_analysis['percentiles']['95th']}")
        
        # Create base prompt to estimate overhead
        sample_prompt = Prompts.batch_analysis_prompt(
            texts[:1], existing_themes, self.batch_config.max_batch_size
        )
        base_prompt_tokens = self.token_counter.count_tokens(sample_prompt)
        
        # Create batches using token-aware algorithm
        batches = []
        current_batch = Batch()
        
        # Sort texts by token count for better packing
        text_items = [
            (i, text, self.token_counter.count_tokens(text))
            for i, text in enumerate(texts)
        ]
        text_items.sort(key=lambda x: x[2])  # Sort by token count
        
        for original_index, text, token_count in text_items:
            # Check if current batch can accommodate this text
            estimated_prompt_tokens = self.token_counter.estimate_prompt_tokens(
                current_batch.texts + [text],
                sample_prompt,
                self.batch_config.token_overhead_per_entry
            )
            
            if (estimated_prompt_tokens > self.batch_config.max_tokens_per_batch and 
                current_batch.texts):
                # Finalize current batch
                current_batch.estimated_tokens = self.token_counter.estimate_prompt_tokens(
                    current_batch.texts,
                    sample_prompt,
                    self.batch_config.token_overhead_per_entry
                )
                batches.append(current_batch)
                current_batch = Batch()
            
            # Add text to current batch
            current_batch.add_text(text, original_index, token_count)
            
            # Check if batch is full
            if len(current_batch.texts) >= self.batch_config.max_batch_size:
                current_batch.estimated_tokens = self.token_counter.estimate_prompt_tokens(
                    current_batch.texts,
                    sample_prompt,
                    self.batch_config.token_overhead_per_entry
                )
                batches.append(current_batch)
                current_batch = Batch()
        
        # Add final batch if it has content
        if current_batch.texts:
            current_batch.estimated_tokens = self.token_counter.estimate_prompt_tokens(
                current_batch.texts,
                sample_prompt,
                self.batch_config.token_overhead_per_entry
            )
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} optimized batches")
        
        # Log batch statistics
        batch_sizes = [len(batch.texts) for batch in batches]
        token_estimates = [batch.estimated_tokens for batch in batches]
        
        logger.info(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, "
                   f"mean={sum(batch_sizes)/len(batch_sizes):.1f}")
        logger.info(f"Token estimates: min={min(token_estimates)}, max={max(token_estimates)}, "
                   f"mean={sum(token_estimates)/len(token_estimates):.1f}")
        
        return batches
    
    def process_batch_sequential(
        self,
        batches: List[Batch],
        processor_function: Callable[[Batch], BatchResult],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchResult]:
        """Process batches sequentially."""
        results = []
        start_time = time.time()
        
        for i, batch in enumerate(batches):
            try:
                batch.mark_processing()
                result = processor_function(batch)
                results.append(result)
                
                # Update statistics
                self.total_batches_processed += 1
                self.total_texts_processed += len(batch.texts)
                
                if progress_callback:
                    progress_callback(i + 1, len(batches))
                
                logger.debug(f"Processed batch {i+1}/{len(batches)} "
                           f"({len(batch.texts)} texts, {result.tokens_used} tokens)")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i+1}: {str(e)}")
                batch.mark_failed(str(e))
                
                # Create error result
                error_result = BatchResult(batch_id=batch.batch_id)
                error_result.errors.append(str(e))
                error_result.mark_completed()
                results.append(error_result)
                
                self.failed_batches += 1
        
        self.total_processing_time += time.time() - start_time
        return results
    
    def process_batch_parallel(
        self,
        batches: List[Batch],
        processor_function: Callable[[Batch], BatchResult],
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchResult]:
        """Process batches in parallel using ThreadPoolExecutor."""
        if max_workers is None:
            max_workers = min(self.max_concurrent_batches, len(batches))
        
        results = [None] * len(batches)  # Maintain order
        completed = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_index = {}
            for i, batch in enumerate(batches):
                batch.mark_processing()
                future = executor.submit(processor_function, batch)
                future_to_index[future] = i
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                batch_index = future_to_index[future]
                
                try:
                    result = future.result()
                    results[batch_index] = result
                    
                    # Update statistics
                    self.total_batches_processed += 1
                    self.total_texts_processed += len(batches[batch_index].texts)
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {batch_index+1}: {str(e)}")
                    batches[batch_index].mark_failed(str(e))
                    
                    # Create error result
                    error_result = BatchResult(batch_id=batches[batch_index].batch_id)
                    error_result.errors.append(str(e))
                    error_result.mark_completed()
                    results[batch_index] = error_result
                    
                    self.failed_batches += 1
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(batches))
                
                logger.debug(f"Completed batch {completed}/{len(batches)}")
        
        self.total_processing_time += time.time() - start_time
        return results
    
    async def process_batch_async(
        self,
        batches: List[Batch],
        async_processor_function: Callable[[Batch], Any],
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchResult]:
        """Process batches asynchronously."""
        if max_concurrent is None:
            max_concurrent = self.max_concurrent_batches
        
        results = []
        completed = 0
        start_time = time.time()
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_batch(batch: Batch, index: int) -> Tuple[int, BatchResult]:
            async with semaphore:
                try:
                    batch.mark_processing()
                    result = await async_processor_function(batch)
                    
                    # Update statistics
                    self.total_batches_processed += 1
                    self.total_texts_processed += len(batch.texts)
                    
                    return index, result
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {index+1}: {str(e)}")
                    batch.mark_failed(str(e))
                    
                    # Create error result
                    error_result = BatchResult(batch_id=batch.batch_id)
                    error_result.errors.append(str(e))
                    error_result.mark_completed()
                    
                    self.failed_batches += 1
                    return index, error_result
        
        # Create tasks for all batches
        tasks = [process_single_batch(batch, i) for i, batch in enumerate(batches)]
        
        # Process tasks and maintain order
        ordered_results = [None] * len(batches)
        
        for coro in asyncio.as_completed(tasks):
            index, result = await coro
            ordered_results[index] = result
            
            completed += 1
            if progress_callback:
                progress_callback(completed, len(batches))
        
        self.total_processing_time += time.time() - start_time
        return ordered_results
    
    def retry_failed_batches(
        self,
        failed_batches: List[Batch],
        processor_function: Callable[[Batch], BatchResult],
        max_retries: int = 2
    ) -> List[BatchResult]:
        """Retry processing failed batches."""
        results = []
        
        for batch in failed_batches:
            if not batch.can_retry():
                logger.warning(f"Batch {batch.batch_id} has exceeded retry limit")
                continue
            
            retry_count = 0
            while retry_count < max_retries and batch.can_retry():
                try:
                    batch.reset_for_retry()
                    batch.mark_processing()
                    
                    # Add delay between retries
                    if retry_count > 0:
                        time.sleep(min(retry_count * 2, 10))  # Exponential backoff
                    
                    result = processor_function(batch)
                    results.append(result)
                    
                    logger.info(f"Successfully retried batch {batch.batch_id} "
                              f"(attempt {retry_count + 1})")
                    break
                    
                except Exception as e:
                    logger.warning(f"Retry {retry_count + 1} failed for batch {batch.batch_id}: {e}")
                    batch.mark_failed(str(e))
                    retry_count += 1
            
            if batch.processing_status == "failed":
                logger.error(f"Batch {batch.batch_id} failed after all retries")
                error_result = BatchResult(batch_id=batch.batch_id)
                error_result.errors.append(f"Failed after {retry_count} retries")
                error_result.mark_completed()
                results.append(error_result)
        
        return results
    
    def optimize_batch_config(
        self,
        sample_texts: List[str],
        target_processing_time: float = 30.0,  # seconds per batch
        max_tokens_per_batch: int = 4000
    ) -> BatchConfig:
        """
        Optimize batch configuration based on sample texts.
        
        Args:
            sample_texts: Sample of texts to analyze
            target_processing_time: Target processing time per batch
            max_tokens_per_batch: Maximum tokens per batch
            
        Returns:
            Optimized BatchConfig
        """
        if not sample_texts:
            return self.batch_config
        
        # Analyze sample texts
        token_analysis = self.token_counter.analyze_token_distribution(sample_texts[:100])
        
        # Estimate optimal batch size based on token usage
        avg_tokens_per_text = token_analysis["mean_tokens"]
        estimated_overhead = 200  # Base prompt overhead
        
        # Calculate optimal batch size
        available_tokens = max_tokens_per_batch - estimated_overhead
        optimal_batch_size = max(1, int(available_tokens / (avg_tokens_per_text + 50)))
        
        # Adjust based on text complexity (longer texts might need smaller batches)
        if token_analysis["percentiles"]["95th"] > 500:  # Long texts
            optimal_batch_size = max(1, optimal_batch_size // 2)
        elif token_analysis["percentiles"]["95th"] < 100:  # Short texts
            optimal_batch_size = min(optimal_batch_size * 2, 25)
        
        # Create optimized config
        optimized_config = BatchConfig(
            max_batch_size=min(optimal_batch_size, 20),
            max_tokens_per_batch=max_tokens_per_batch,
            target_tokens_per_batch=int(max_tokens_per_batch * 0.8),
            token_overhead_per_entry=max(50, int(avg_tokens_per_text * 0.1)),
            enable_dynamic_sizing=True
        )
        
        logger.info(f"Optimized batch config: max_size={optimized_config.max_batch_size}, "
                   f"target_tokens={optimized_config.target_tokens_per_batch}")
        
        return optimized_config
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        avg_processing_time = (
            self.total_processing_time / self.total_batches_processed
            if self.total_batches_processed > 0 else 0
        )
        
        avg_texts_per_batch = (
            self.total_texts_processed / self.total_batches_processed
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
            "average_texts_per_batch": avg_texts_per_batch,
            "processing_speed": {
                "texts_per_second": (
                    self.total_texts_processed / self.total_processing_time
                    if self.total_processing_time > 0 else 0
                ),
                "batches_per_minute": (
                    (self.total_batches_processed / self.total_processing_time) * 60
                    if self.total_processing_time > 0 else 0
                )
            },
            "current_config": {
                "max_batch_size": self.batch_config.max_batch_size,
                "max_tokens_per_batch": self.batch_config.max_tokens_per_batch,
                "target_tokens_per_batch": self.batch_config.target_tokens_per_batch
            }
        }