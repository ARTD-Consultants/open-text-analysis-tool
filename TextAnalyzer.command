#!/bin/bash
cd "$(dirname "$0")"
echo "🤖 Text Analyzer - Ready!"
echo "======================="
echo ""
echo "Quick commands:"
echo "python3 run_analyzer.py preview application_open_text.xlsx"
echo "python3 run_analyzer.py analyze application_open_text.xlsx --text-column 'YourColumn'"
echo ""
echo "Type your commands below (or 'exit' to quit):"
echo ""
exec bash