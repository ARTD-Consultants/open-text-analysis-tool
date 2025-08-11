# Qualitative Text Analyzer

An AI-powered framework for analyzing open-text survey responses and qualitative data using Azure OpenAI GPT models.

## Features

- **Intelligent Batch Processing**: Dynamic batch sizing based on text length and token usage
- **Theme Consistency**: Automatic theme validation and merging using semantic similarity
- **Hierarchical Themes**: Organize themes in parent-child relationships
- **Caching System**: Intelligent caching to reduce API costs and processing time  
- **Comprehensive Reports**: Generate Word documents, Excel summaries, and visualizations
- **Progress Tracking**: Real-time progress monitoring with detailed statistics
- **Quality Validation**: Input validation and data quality assessment

## Installation

### Option 1: Easy Installation (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qualitative-text-analyzer.git
cd qualitative-text-analyzer
```

2. Run the installer:
```bash
./install.sh
```

3. Configure your API keys:
```bash
# Edit .env with your Azure OpenAI credentials
cp .env.example .env
nano .env  # or use your preferred editor
```

That's it! You can now use `qualitative-analyzer` from anywhere on your system.

### Option 2: Manual Setup

1. Clone and navigate:
```bash
git clone https://github.com/yourusername/qualitative-text-analyzer.git
cd qualitative-text-analyzer
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your actual API keys and endpoints
```

3. Use the local runner:
```bash
python run_analyzer.py [commands]
```

## Quick Start

### Easy Method (After Installation)

Once installed, you can run the analyzer from **anywhere**:

```bash
# Basic analysis
qualitative-analyzer analyze survey.xlsx --text-column "responses"

# With comprehensive report
qualitative-analyzer analyze survey.xlsx --text-column "responses" --report

# Preview your data first
qualitative-analyzer preview survey.xlsx

# Test your API connection
qualitative-analyzer test-connection

# Get help
qualitative-analyzer --help
```

### Manual Method (Without Installation)

If you prefer not to install globally, run from the project directory:

```bash
# Navigate to project directory first
cd /path/to/qualitative-text-analyzer

# Then run commands
python run_analyzer.py analyze data.xlsx --text-column "response"
python run_analyzer.py preview data.xlsx
python run_analyzer.py test-connection
```

### Python API

```python
from qualitative_analyzer import QualitativeAnalyzer

# Initialize analyzer from environment
analyzer = QualitativeAnalyzer.from_env()

# Analyze file
session = analyzer.analyze_file(
    input_file="survey_data.xlsx",
    text_column="response",
    output_file="results.xlsx"
)

# Generate comprehensive report
report_files = analyzer.generate_comprehensive_report()
print("Reports generated:", report_files)

# Get analysis summary
summary = analyzer.get_analysis_summary()
print("Top themes:", summary['top_themes'])
```

## Configuration

The system uses environment variables for configuration. Copy `.env.example` to `.env` and configure:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-2
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# Analysis Configuration
DEFAULT_BATCH_SIZE=15
MAX_TOKENS=4000
API_TEMPERATURE=0.5
THEME_SIMILARITY_THRESHOLD=0.85

# Features
ENABLE_THEME_SIMILARITY=true
ENABLE_CACHING=true
GENERATE_THEME_REPORT=true
```

## Input File Requirements

- **Supported formats**: Excel (.xlsx, .xls), CSV (.csv)
- **Required**: At least one column containing text data
- **Recommended**: Text entries between 10-1000 characters for optimal results

## Output Files

The analyzer generates multiple output formats:

1. **Excel Summary** (`analysis_summary_TIMESTAMP.xlsx`):
   - Analysis results with themes and summaries
   - Theme frequency statistics
   - Session statistics
   - Theme co-occurrence analysis

2. **Word Report** (`detailed_report_TIMESTAMP.docx`):
   - Comprehensive thematic analysis
   - AI-generated theme descriptions
   - Representative quotes
   - Visualizations

3. **Visualizations** (`theme_charts_TIMESTAMP.png`):
   - Theme frequency charts
   - Confidence vs frequency plots
   - Distribution analyses

## Advanced Features

### Theme Management

```python
# Get theme merge suggestions
suggestions = analyzer.suggest_theme_merges(max_suggestions=10)

# Manually merge themes
analyzer.theme_manager.merge_themes("Career Change", "Job Transition")

# Create theme hierarchy
analyzer.theme_manager.create_theme_hierarchy(
    "Career Challenges", 
    ["Job Search", "Career Change", "Skill Development"]
)
```

### Caching and Performance

```python
# Get cache statistics
cache_stats = analyzer.cache_manager.get_cache_statistics()
print(f"Cache hit rate: {cache_stats['cache_hit_rate']:.1%}")

# Clear cache
analyzer.cache_manager.clear_cache()
```

### Quality Assessment

```python
# Validate theme quality
quality_report = analyzer.theme_manager.validate_theme_quality()
print(f"Theme quality: {quality_report['status']}")
```

## Architecture

The framework is built with a modular architecture:

```
qualitative_analyzer/
├── config/          # Configuration management
├── core/            # Core business logic
├── services/        # External service integrations
├── models/          # Data models and structures
├── utils/           # Utility functions
└── main.py          # CLI interface
```

## Best Practices

1. **Data Preparation**: Clean your text data before analysis
2. **Batch Sizing**: Let the system optimize batch sizes automatically
3. **Theme Validation**: Review suggested theme merges before finalizing
4. **Caching**: Enable caching to reduce costs on repeated analyses
5. **Quality Checks**: Use the validation commands to assess data quality

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify Azure OpenAI credentials and endpoints
2. **Token Limits**: Reduce batch size or text length for very long texts
3. **Memory Issues**: Process large datasets in smaller chunks
4. **Theme Inconsistency**: Enable theme similarity checking

### Getting Help

```bash
# Test your configuration
python -m qualitative_analyzer config test

# Validate your input file
python -m qualitative_analyzer validate data.xlsx --text-column response

# View current configuration
python -m qualitative_analyzer config show
```

## Development

### Running Tests

```bash
pip install -e .[dev]
pytest tests/
```

### Code Formatting

```bash
black qualitative_analyzer/
isort qualitative_analyzer/
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Citation

If you use this framework in academic research, please cite:

```bibtex
@software{qualitative_text_analyzer,
  title={Qualitative Text Analyzer: AI-Powered Thematic Analysis Framework},
  author={Your Team},
  year={2024},
  url={https://github.com/yourusername/qualitative-text-analyzer}
}
```