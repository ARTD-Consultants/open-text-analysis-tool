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
        max_words: int = 1000
    ) -> str:
        """Create prompt for detailed theme analysis."""
        
        # Limit examples to avoid token overflow
        max_examples = min(30, len(text_entries))
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
        max_quotes: int = 5
    ) -> str:
        """Create prompt for extracting representative quotes."""
        
        # Limit entries to manage token usage
        sample_size = min(30, len(text_entries))
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
    def confidence_scoring_prompt(texts: List[str]) -> str:
        """Create prompt that includes confidence scoring."""
        
        entries = ""
        for i, text in enumerate(texts, 1):
            entries += f"{i}: {text}\\n"
        
        return f"""Analyze texts for themes with confidence scores (0-100%).

Format: Summary|Theme1(confidence%),Theme2(confidence%)

Texts:
{entries}

Output:
1: Summary here|Theme1(85%),Theme2(92%)
2: Summary here|Theme1(78%)
..."""

    @staticmethod
    def representative_themes_creation_prompt(
        original_themes: list, 
        num_representative_themes: int = 10
    ) -> str:
        """Create prompt for generating representative themes from original themes."""
        
        theme_count = len(original_themes)
        themes_text = ', '.join(original_themes)
        
        return f"""Analyze these {theme_count} themes and create exactly {num_representative_themes} broad, inclusive themes that capture their diversity:

{themes_text}

Create {num_representative_themes} comprehensive categories that most themes would fit into. Make them:
1. Broad enough to capture multiple similar themes
2. Specific enough to be meaningful
3. Cover the full range of concepts present in the data

Return only the {num_representative_themes} theme names, one per line."""

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