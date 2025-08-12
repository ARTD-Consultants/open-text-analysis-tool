"""Azure OpenAI client wrapper with retry logic and embedding support."""

import time
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from openai import AzureOpenAI
import backoff

from ..config.prompts import Prompts

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for Azure OpenAI API with retry logic and embedding support."""
    
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        embedding_deployment_name: str,
        settings
    ):
        """Initialize the OpenAI client."""
        self.settings = settings
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=settings.openai_timeout
        )
        
        self.deployment_name = deployment_name
        self.embedding_deployment_name = embedding_deployment_name
        
        # Statistics tracking
        self.total_requests = 0
        self.total_tokens_used = 0
        self.failed_requests = 0
    
    def _make_chat_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """Make a chat completion request with retry logic."""
        # Use settings defaults if not provided
        if temperature is None:
            temperature = self.settings.api_temperature
        if max_tokens is None:
            max_tokens = self.settings.max_tokens
            
        for attempt in range(self.settings.openai_max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                # Update statistics
                self.total_requests += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens_used += response.usage.total_tokens
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == self.settings.openai_max_retries - 1:
                    # Last attempt failed
                    self.failed_requests += 1
                    logger.error(f"Chat completion request failed after {self.settings.openai_max_retries} attempts: {str(e)}")
                    raise
                else:
                    # Wait before retrying
                    wait_time = self.settings.openai_retry_delay * (self.settings.backoff_factor ** attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.settings.openai_max_retries}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
    
    def analyze_text_batch(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Analyze a batch of texts using chat completion."""
        messages = [{"role": "user", "content": prompt}]
        
        return self._make_chat_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def generate_theme_analysis(
        self,
        theme: str,
        examples: List[str],
        total_count: int,
        max_words: int = None
    ) -> str:
        """Generate detailed analysis for a specific theme."""
        if max_words is None:
            max_words = self.settings.theme_analysis_max_words
        prompt = Prompts.theme_analysis_prompt(
            theme, examples, total_count, max_words, 
            max_examples=self.settings.max_theme_examples
        )
        return self.analyze_text_batch(prompt, max_tokens=max_words * 2)
    
    def extract_representative_quotes(
        self,
        theme: str,
        examples: List[str],
        max_quotes: int = None
    ) -> List[str]:
        """Extract representative quotes for a theme."""
        if max_quotes is None:
            max_quotes = self.settings.max_representative_quotes
        prompt = Prompts.quote_extraction_prompt(
            theme, examples, max_quotes,
            max_examples=self.settings.max_theme_examples
        )
        response = self.analyze_text_batch(prompt)
        
        # Parse quotes from response
        quotes = []
        for line in response.split('\\n'):
            line = line.strip()
            if line.startswith('QUOTE') and '"' in line:
                # Extract text between quotes
                start = line.find('"')
                end = line.rfind('"')
                if start != -1 and end != -1 and start != end:
                    quote = line[start + 1:end]
                    quotes.append(quote)
        
        return quotes[:max_quotes]
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_deployment_name,
                input=text
            )
            
            self.total_requests += 1
            return response.data[0].embedding
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Embedding request failed: {str(e)}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts efficiently."""
        # Azure OpenAI supports batch embedding requests
        try:
            response = self.client.embeddings.create(
                model=self.embedding_deployment_name,
                input=texts
            )
            
            self.total_requests += 1
            embeddings = []
            
            for data in response.data:
                embeddings.append(data.embedding)
            
            return embeddings
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Batch embedding request failed: {str(e)}")
            
            # Fallback to individual requests
            return [self.get_embedding(text) for text in texts]
    
    def validate_theme_similarity(
        self,
        new_theme: str,
        existing_theme: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """Use GPT to validate if two themes are similar."""
        prompt = Prompts.theme_validation_prompt(new_theme, existing_theme, context or "General analysis")
        
        try:
            response = self.analyze_text_batch(prompt, max_tokens=200)
            
            # Parse response
            result = {
                "similar": False,
                "confidence": 0,
                "reason": ""
            }
            
            for line in response.split('\\n'):
                line = line.strip()
                if line.startswith('Similar:'):
                    result["similar"] = 'yes' in line.lower()
                elif line.startswith('Confidence:'):
                    try:
                        confidence_str = line.split(':')[1].strip().replace('%', '')
                        result["confidence"] = int(confidence_str)
                    except:
                        pass
                elif line.startswith('Reason:'):
                    result["reason"] = line.split(':', 1)[1].strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Theme validation failed: {str(e)}")
            return {"similar": False, "confidence": 0, "reason": "Validation failed"}
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the client."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "failed_requests": self.failed_requests,
            "success_rate": (
                (self.total_requests - self.failed_requests) / self.total_requests 
                if self.total_requests > 0 else 0
            ),
            "average_tokens_per_request": (
                self.total_tokens_used / self.total_requests 
                if self.total_requests > 0 else 0
            )
        }
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        input_cost_per_1k: float = 0.005,
        output_cost_per_1k: float = 0.015
    ) -> float:
        """Estimate cost for token usage."""
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        return input_cost + output_cost
    
    def consolidate_themes(
        self,
        original_themes: List[str],
        final_theme_count: int = None,
        consolidation_deployment: str = None
    ) -> List[str]:
        """Consolidate original themes into a smaller set using GPT-4."""
        if final_theme_count is None:
            final_theme_count = self.settings.final_theme_count
        if consolidation_deployment is None:
            consolidation_deployment = self.settings.consolidation_deployment
            
        prompt = Prompts.theme_consolidation_prompt(original_themes, final_theme_count)
        
        # Use specific deployment for consolidation (typically GPT-4)
        original_deployment = self.deployment_name
        self.deployment_name = consolidation_deployment
        
        try:
            response = self.analyze_text_batch(
                prompt=prompt,
                temperature=self.settings.consolidation_temperature,
                max_tokens=self.settings.consolidation_max_tokens
            )
            
            # Parse consolidated themes from response
            consolidated_themes = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('- ')):
                    # Remove numbering and bullet points
                    theme_name = line
                    if '. ' in theme_name:
                        theme_name = theme_name.split('. ', 1)[1]
                    elif '- ' in theme_name:
                        theme_name = theme_name.replace('- ', '')
                    
                    theme_name = theme_name.strip()
                    if theme_name and theme_name not in consolidated_themes:
                        consolidated_themes.append(theme_name)
            
            # Ensure we have the expected number of themes
            if len(consolidated_themes) < final_theme_count:
                logger.warning(f"Only got {len(consolidated_themes)} consolidated themes, expected {final_theme_count}")
            
            return consolidated_themes[:final_theme_count]
            
        except Exception as e:
            logger.error(f"Theme consolidation failed: {str(e)}")
            # Fallback: return most frequent original themes
            from collections import Counter
            theme_counts = Counter(original_themes)
            return [theme for theme, _ in theme_counts.most_common(final_theme_count)]
        
        finally:
            # Restore original deployment
            self.deployment_name = original_deployment
    
