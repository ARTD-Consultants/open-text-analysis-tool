"""Centralized prompt templates for consistent analysis."""

from typing import List


class Prompts:
    """Collection of all prompt templates used in analysis."""
    
    @staticmethod
    def batch_analysis_prompt(
        text_list: List[str], 
        existing_themes: List[str] = None,
        max_themes: int = 3
    ) -> str:
        """Create optimized prompt for batch text analysis."""
        
        # Build theme guidance
        theme_guidance = ""
        if existing_themes:
            theme_guidance = f"\\nKnown themes: {', '.join(existing_themes)}\\nUse existing themes when appropriate, create new ones when needed."
        
        # Build text entries
        entries = ""
        for i, text in enumerate(text_list, 1):
            entries += f"{i}: {text}\\n"
        
        return f"""Analyze texts for themes and summaries. Output format: Summary|Theme1,Theme2,Theme3
        
Max {max_themes} themes per text. Keep theme names concise (1-3 words).{theme_guidance}

Texts:
{entries}

Format:
1: Brief summary here|Theme1,Theme2
2: Brief summary here|Theme1
..."""

    @staticmethod
    def theme_analysis_prompt(
        theme: str, 
        text_entries: List[str], 
        total_entries: int,
        max_words: int = 1000,
        max_examples: int = 30
    ) -> str:
        """Create prompt for detailed theme analysis."""
        
        # Limit examples to avoid token overflow
        max_examples = min(max_examples, len(text_entries))
        examples = text_entries[:max_examples]
        remaining_count = total_entries - max_examples
        
        examples_text = "\\n".join([f'- "{example}"' for example in examples])
        remaining_text = f"\\n(Plus {remaining_count} more entries not shown)" if remaining_count > 0 else ""
        
        return f"""Analyze the theme "{theme}" based on text entries.

THEME: {theme}
TOTAL ENTRIES: {total_entries}

EXAMPLES ({max_examples} of {total_entries}):
{examples_text}{remaining_text}

Write a {max_words}-word analysis covering:
1. Key aspects and dimensions of this theme
2. Patterns and variations within the theme  
3. Significance and implications

Base analysis only on the provided data. Do not make unsupported claims."""

    @staticmethod
    def quote_extraction_prompt(
        theme: str,
        text_entries: List[str],
        max_quotes: int = 5,
        max_examples: int = 30
    ) -> str:
        """Create prompt for extracting representative quotes."""
        
        # Limit entries to manage token usage
        sample_size = min(max_examples, len(text_entries))
        sample_entries = text_entries[:sample_size]
        
        entries_text = "\\n".join([f'- "{entry}"' for entry in sample_entries])
        
        return f"""Extract {max_quotes} most representative quotes for theme "{theme}".

THEME: {theme}
TEXT ENTRIES:
{entries_text}

Select quotes that:
1. Clearly illustrate the theme
2. Show different aspects/variations
3. Are verbatim from entries (no paraphrasing)

Format:
QUOTE 1: "exact quote text"
QUOTE 2: "exact quote text"
..."""

    @staticmethod
    def theme_validation_prompt(
        new_theme: str,
        existing_theme: str,
        context: str
    ) -> str:
        """Create prompt for validating theme similarity."""
        
        return f"""Are these themes the same concept in this context?

New theme: "{new_theme}"
Existing theme: "{existing_theme}"
Context: {context}

Answer: Yes/No
Confidence: 0-100%
Reason: Brief explanation"""


    @staticmethod
    def theme_mapping_prompt(
        original_themes_chunk: list, 
        representative_themes: list
    ) -> str:
        """Create prompt for mapping original themes to representative themes."""
        
        representative_list = ', '.join(representative_themes)
        original_list = ', '.join(original_themes_chunk)
        
        return f"""Map each of these original themes to the BEST matching representative theme:

REPRESENTATIVE THEMES: {representative_list}

ORIGINAL THEMES TO MAP:
{original_list}

Format your response as:
Original Theme 1 -> Representative Theme
Original Theme 2 -> Representative Theme
...

Use the exact representative theme names provided above."""

    @staticmethod
    def theme_consolidation_prompt(
        original_themes: List[str],
        final_theme_count: int = 10
    ) -> str:
        """Create comprehensive prompt for consolidating themes using GPT-4."""
        
        themes_text = '\n'.join([f"- {theme}" for theme in original_themes])
        theme_count = len(original_themes)
        
        return f"""You are analyzing qualitative research data. Below are {theme_count} themes that were generated from text analysis. Your task is to consolidate these into exactly {final_theme_count} broader, more meaningful themes.

ORIGINAL THEMES ({theme_count} total):
{themes_text}

INSTRUCTIONS:
1. Create exactly {final_theme_count} consolidated themes
2. Each consolidated theme should:
   - Capture multiple related original themes
   - Be broad enough to be meaningful but specific enough to be actionable
   - Use clear, descriptive names (2-4 words)
   - Cover the full conceptual range of the data

3. Ensure all major concepts from the original themes are represented
4. Prioritize themes that appear most frequently or are most significant

RESPONSE FORMAT:
Provide exactly {final_theme_count} consolidated theme names, one per line:
1. [Consolidated Theme Name]
2. [Consolidated Theme Name]
...
{final_theme_count}. [Consolidated Theme Name]

Only return the numbered list of consolidated theme names."""