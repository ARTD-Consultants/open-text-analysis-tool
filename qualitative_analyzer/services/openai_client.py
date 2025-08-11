"""Azure OpenAI client wrapper with retry logic and embedding support."""

import time
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from openai import AzureOpenAI
import backoff

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for Azure OpenAI API with retry logic and embedding support."""
    
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2024-10-21",
        deployment_name: str = "gpt-4o-2",
        embedding_deployment_name: str = "text-embedding-ada-002",
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: int = 60
    ):
        """Initialize the OpenAI client."""
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout
        )
        
        self.deployment_name = deployment_name
        self.embedding_deployment_name = embedding_deployment_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Statistics tracking
        self.total_requests = 0
        self.total_tokens_used = 0
        self.failed_requests = 0
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        factor=2
    )
    def _make_chat_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 4000,
        **kwargs
    ) -> str:
        """Make a chat completion request with retry logic."""
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
            self.failed_requests += 1
            logger.error(f"Chat completion request failed: {str(e)}")
            raise
    
    def analyze_text_batch(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 4000
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
        max_words: int = 1000
    ) -> str:
        """Generate detailed analysis for a specific theme."""
        examples_text = "\\n".join([f'- "{example}"' for example in examples[:30]])
        remaining = max(0, total_count - len(examples[:30]))
        
        prompt = f"""Analyze the theme "{theme}" based on {total_count} text entries.

THEME: {theme}

EXAMPLES ({len(examples[:30])} of {total_count}):
{examples_text}
{f'(Plus {remaining} more entries)' if remaining > 0 else ''}

Write a {max_words}-word analysis covering:
1. Key aspects and dimensions of this theme
2. Patterns and variations within the theme
3. Significance and implications

Base analysis only on the provided data."""
        
        return self.analyze_text_batch(prompt, max_tokens=max_words * 2)
    
    def extract_representative_quotes(
        self,
        theme: str,
        examples: List[str],
        max_quotes: int = 5
    ) -> List[str]:
        """Extract representative quotes for a theme."""
        examples_text = "\\n".join([f'- "{example}"' for example in examples[:30]])
        
        prompt = f"""Extract {max_quotes} most representative quotes for theme "{theme}".

THEME: {theme}
EXAMPLES:
{examples_text}

Select quotes that clearly illustrate the theme and show different aspects.
Extract verbatim text only.

Format:
QUOTE 1: "exact quote text"
QUOTE 2: "exact quote text"
..."""
        
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
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        factor=2
    )
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
        prompt = f"""Are these themes the same concept?

New theme: "{new_theme}"
Existing theme: "{existing_theme}"
{f'Context: {context}' if context else ''}

Answer with:
- Similar: Yes/No
- Confidence: 0-100%
- Reason: Brief explanation

Format:
Similar: Yes
Confidence: 85%
Reason: Both refer to career transitions"""
        
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
    
    def test_connection(self) -> bool:
        """Test the connection to Azure OpenAI."""
        try:
            response = self.analyze_text_batch(
                "Test connection. Respond with 'OK'.",
                max_tokens=10
            )
            return "ok" in response.lower()
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False