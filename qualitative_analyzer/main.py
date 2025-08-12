"""Main CLI interface for the Qualitative Text Analyzer."""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

from qualitative_analyzer.core.analyzer import QualitativeAnalyzer
from qualitative_analyzer.config.settings import Settings
from qualitative_analyzer.utils.validators import validate_input_file


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Qualitative Text Analyzer - AI-powered thematic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  analyze data.xlsx --text-column "response"
  
  # With custom output
  analyze data.xlsx -t response -o results.xlsx
  
  # Generate comprehensive report
  analyze data.xlsx -t response --report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze text data')
    analyze_parser.add_argument('input_file', help='Input file path (.xlsx, .xls, .csv)')
    analyze_parser.add_argument('-t', '--text-column', 
                               help='Column containing text to analyze')
    analyze_parser.add_argument('-o', '--output', 
                               help='Output file path (optional)')
    analyze_parser.add_argument('-s', '--sheet', 
                               help='Excel sheet name (optional)')
    analyze_parser.add_argument('--report', action='store_true',
                               help='Generate comprehensive report')
    analyze_parser.add_argument('--report-title', default='Qualitative Analysis Report',
                               help='Title for the report')
    analyze_parser.add_argument('--no-cache', action='store_true',
                               help='Disable caching')
    analyze_parser.add_argument('--no-similarity', action='store_true',
                               help='Disable theme similarity checking')
    analyze_parser.add_argument('--final-theme-count', type=int, default=None,
                               help='Number of final consolidated themes (uses .env setting if not specified)')
    analyze_parser.add_argument('--no-consolidation', action='store_true',
                               help='Disable GPT-4 theme consolidation')
    
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    show_config_parser = config_subparsers.add_parser('show', help='Show current configuration')
    test_config_parser = config_subparsers.add_parser('test', help='Test configuration')
    
    # Global options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--env-file', help='Environment file path')
    
    return parser


def command_analyze(args, settings: Settings):
    """Handle analyze command."""
    print(f"Starting analysis of {args.input_file}...")
    
    # Validate input file
    required_columns = [args.text_column] if args.text_column else [settings.default_text_column]
    is_valid, error_msg = validate_input_file(args.input_file, required_columns)
    
    if not is_valid:
        print(f"❌ Input validation failed: {error_msg}")
        return 1
    
    try:
        # Override settings with command-line arguments (only if explicitly provided)
        if hasattr(args, 'final_theme_count') and args.final_theme_count is not None:
            settings.final_theme_count = args.final_theme_count
        if hasattr(args, 'no_consolidation'):
            settings.enable_theme_consolidation = not args.no_consolidation
        
        # Create analyzer
        analyzer = QualitativeAnalyzer(
            settings=settings,
            enable_caching=not args.no_cache,
            enable_theme_similarity=not args.no_similarity
        )
        
        
        # Progress callback
        def progress_callback(completed: int, total: int):
            percent = (completed / total) * 100
            print(f"Progress: {completed}/{total} batches ({percent:.1f}%)")
        
        # Run analysis
        session = analyzer.analyze_file(
            input_file=args.input_file,
            text_column=args.text_column,
            output_file=args.output,
            sheet_name=args.sheet,
            progress_callback=progress_callback
        )
        
        # Generate report if requested
        if args.report:
            print("\\n📄 Generating comprehensive report...")
            report_files = analyzer.generate_simple_report(args.report_title)
            
            print("Reports generated:")
            for report_type, file_path in report_files.items():
                print(f"  • {report_type.capitalize()}: {file_path}")
        
        if args.output:
            print(f"\\n✅ Results saved to: {args.output}")
        
        print("\\n🎉 Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        logging.error(f"Analysis error: {str(e)}", exc_info=True)
        return 1




def command_config(args, settings: Settings):
    """Handle config command."""
    if args.config_action == 'show':
        print("⚙️  Current Configuration:")
        print(f"  • OpenAI Endpoint: {settings.azure_openai_endpoint}")
        print(f"  • Deployment Name: {settings.azure_openai_deployment_name}")
        print(f"  • Embedding Deployment: {settings.azure_openai_embedding_deployment_name}")
        print(f"  • Batch Size: {settings.batch_size}")
        print(f"  • Max Tokens: {settings.max_tokens}")
        print(f"  • API Temperature: {settings.api_temperature}")
        print(f"  • Similarity Threshold: {settings.similarity_threshold}")
        return 0
        
    elif args.config_action == 'test':
        print("🧪 Testing configuration...")
        try:
            settings.validate()
            print("✅ Configuration is valid!")
            return 0
        except Exception as e:
            print(f"❌ Configuration error: {str(e)}")
            return 1
    
    return 0


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        # Load settings
        settings = Settings.from_env(args.env_file)
        
        # Route to appropriate command handler
        if args.command == 'analyze':
            return command_analyze(args, settings)
        elif args.command == 'config':
            return command_config(args, settings)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logging.error(f"Main error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())