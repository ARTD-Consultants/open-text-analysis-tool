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
  
  # Test connection
  test-connection
  
  # Preview data file
  preview data.xlsx
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
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview data file')
    preview_parser.add_argument('input_file', help='Input file path')
    preview_parser.add_argument('-n', '--rows', type=int, default=5,
                               help='Number of rows to preview')
    preview_parser.add_argument('-t', '--text-column',
                               help='Text column to analyze')
    
    # Test connection command
    test_parser = subparsers.add_parser('test-connection', 
                                       help='Test connection to OpenAI services')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate input file')
    validate_parser.add_argument('input_file', help='Input file path')
    validate_parser.add_argument('-t', '--text-column', required=True,
                                help='Column containing text to analyze')
    
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
        # Create analyzer
        analyzer = QualitativeAnalyzer(
            settings=settings,
            enable_caching=not args.no_cache,
            enable_theme_similarity=not args.no_similarity
        )
        
        # Test connection first
        print("Testing connection to OpenAI services...")
        connection_status = analyzer.test_connection()
        
        if not connection_status['openai_connection']:
            print("❌ Failed to connect to OpenAI. Please check your configuration.")
            return 1
        
        print("✅ Connection successful")
        
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
        
        # Print summary
        stats = session.get_session_statistics()
        print("\\n📊 Analysis Summary:")
        print(f"  • Entries processed: {stats['total_entries']}")
        print(f"  • Unique themes found: {stats['unique_themes']}")
        print(f"  • Processing time: {stats.get('session_duration', 0):.1f} seconds")
        print(f"  • Average themes per entry: {stats['average_themes_per_entry']:.1f}")
        print(f"  • Overall confidence: {stats['overall_confidence']:.1%}")
        
        # Show top themes
        theme_summary = session.get_global_theme_summary()
        print("\\n🏷️  Top Themes:")
        sorted_themes = sorted(theme_summary.items(), key=lambda x: x[1], reverse=True)
        for i, (theme_name, count) in enumerate(sorted_themes[:5], 1):
            print(f"  {i}. {theme_name} ({count} occurrences)")
        
        # Generate report if requested
        if args.report:
            print("\\n📄 Generating comprehensive report...")
            report_files = analyzer.generate_comprehensive_report(args.report_title)
            
            print("Reports generated:")
            for report_type, file_path in report_files.items():
                print(f"  • {report_type.capitalize()}: {file_path}")
        
        # Show suggestions
        suggestions = analyzer.suggest_theme_merges(max_suggestions=5)
        if suggestions:
            print("\\n💡 Theme merge suggestions:")
            for suggestion in suggestions:
                print(f"  • Merge '{suggestion['theme1']}' with '{suggestion['theme2']}' "
                     f"(similarity: {suggestion['similarity']:.2f})")
        
        if args.output:
            print(f"\\n✅ Results saved to: {args.output}")
        
        print("\\n🎉 Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        logging.error(f"Analysis error: {str(e)}", exc_info=True)
        return 1


def command_preview(args, settings: Settings):
    """Handle preview command."""
    try:
        from qualitative_analyzer.services.data_loader import DataLoader
        
        loader = DataLoader()
        preview_info = loader.preview_data(
            args.input_file, 
            args.rows, 
            args.text_column
        )
        
        if 'error' in preview_info:
            print(f"❌ Preview failed: {preview_info['error']}")
            return 1
        
        print(f"📄 File Preview: {args.input_file}")
        print(f"Size: {preview_info['file_size_mb']} MB")
        print(f"Columns ({preview_info['column_count']}): {', '.join(preview_info['columns'])}")
        
        if 'text_analysis' in preview_info:
            text_analysis = preview_info['text_analysis']
            print(f"\\n📝 Text Analysis (column: {args.text_column}):")
            print(f"Average length: {text_analysis['avg_length']:.1f} characters")
            
            print("\\nSample texts:")
            for i, text in enumerate(text_analysis['sample_texts'], 1):
                preview_text = text[:100] + "..." if len(text) > 100 else text
                print(f"  {i}. {preview_text}")
        
        print("\\n🔍 Column Suggestions:")
        suggestions = loader.get_column_suggestions(args.input_file)
        if 'text_columns' in suggestions:
            print(f"Likely text columns: {', '.join(suggestions['text_columns'])}")
        if 'id_columns' in suggestions:
            print(f"Likely ID columns: {', '.join(suggestions['id_columns'])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Preview failed: {str(e)}")
        return 1


def command_test_connection(args, settings: Settings):
    """Handle test-connection command."""
    try:
        print("🔗 Testing connection to OpenAI services...")
        
        analyzer = QualitativeAnalyzer(settings=settings)
        connection_status = analyzer.test_connection()
        
        print("\\nConnection Status:")
        print(f"  • OpenAI API: {'✅ Connected' if connection_status['openai_connection'] else '❌ Failed'}")
        print(f"  • Caching: {'✅ Available' if connection_status['cache_available'] else '❌ Disabled'}")
        print(f"  • Embeddings: {'✅ Available' if connection_status['embeddings_available'] else '❌ Disabled'}")
        
        if connection_status['openai_connection']:
            # Get usage statistics
            stats = analyzer.openai_client.get_usage_statistics()
            print(f"\\n📊 API Statistics:")
            print(f"  • Total requests: {stats['total_requests']}")
            print(f"  • Success rate: {stats['success_rate']:.1%}")
            
            print("\\n🎉 Connection test successful!")
            return 0
        else:
            print("\\n❌ Connection test failed. Please check your configuration.")
            return 1
            
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        return 1


def command_validate(args, settings: Settings):
    """Handle validate command."""
    try:
        print(f"🔍 Validating file: {args.input_file}")
        
        # Basic file validation
        is_valid, error_msg = validate_input_file(args.input_file, [args.text_column])
        
        if not is_valid:
            print(f"❌ Validation failed: {error_msg}")
            return 1
        
        # Load and validate data quality
        from qualitative_analyzer.services.data_loader import DataLoader
        
        loader = DataLoader()
        df, metadata = loader.load_data(args.input_file, args.text_column)
        
        print(f"✅ File validation successful!")
        print(f"\\n📊 Data Quality Report:")
        
        quality = metadata['data_quality']
        print(f"  • Total entries: {quality['total_entries']}")
        print(f"  • Valid entries: {quality['valid_entries']} ({quality['valid_percentage']:.1f}%)")
        print(f"  • Quality score: {quality['quality_score']:.1f}/100")
        
        if quality['missing_entries'] > 0:
            print(f"  • Missing entries: {quality['missing_entries']}")
        if quality['empty_entries'] > 0:
            print(f"  • Empty entries: {quality['empty_entries']}")
        
        # Show recommendations
        if quality['recommendations']:
            print(f"\\n💡 Recommendations:")
            for rec in quality['recommendations']:
                print(f"  • {rec}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return 1


def command_config(args, settings: Settings):
    """Handle config command."""
    if args.config_action == 'show':
        print("⚙️  Current Configuration:")
        print(f"  • OpenAI Endpoint: {settings.azure_openai_endpoint}")
        print(f"  • Deployment Name: {settings.azure_openai_deployment_name}")
        print(f"  • Embedding Deployment: {settings.azure_openai_embedding_deployment_name}")
        print(f"  • Default Batch Size: {settings.default_batch_size}")
        print(f"  • Max Tokens: {settings.max_tokens}")
        print(f"  • API Temperature: {settings.api_temperature}")
        print(f"  • Theme Similarity: {settings.enable_theme_similarity}")
        print(f"  • Caching: {settings.enable_caching}")
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
        elif args.command == 'preview':
            return command_preview(args, settings)
        elif args.command == 'test-connection':
            return command_test_connection(args, settings)
        elif args.command == 'validate':
            return command_validate(args, settings)
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