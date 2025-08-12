#!/usr/bin/env python3
"""Super simple launcher - just put your file in this folder and run!"""

import sys
import subprocess
from pathlib import Path

def find_data_files():
    """Find Excel/CSV files in the current directory."""
    current_dir = Path(__file__).parent
    data_files = []
    
    # Look for common data file extensions
    for ext in ['*.xlsx', '*.xls', '*.csv']:
        data_files.extend(current_dir.glob(ext))
    
    return data_files

def setup_environment():
    """Set up the virtual environment."""
    script_dir = Path(__file__).parent
    venv_path = script_dir / "qualitative-analyzer-env"
    
    # Get python executable
    python_exe = venv_path / "bin" / "python"
    if not python_exe.exists():
        python_exe = venv_path / "Scripts" / "python.exe"  # Windows
    
    if not python_exe.exists():
        print("❌ Virtual environment not found. Please run 'python3 run_analyzer.py --help' first.")
        return None
    
    return str(python_exe)

def main():
    print("🚀 Simple Qualitative Text Analyzer")
    print("=" * 50)
    
    # Find data files
    data_files = find_data_files()
    
    if not data_files:
        print("❌ No data files found!")
        print()
        print("Please copy your Excel (.xlsx, .xls) or CSV (.csv) file")
        print("into this folder, then run this script again.")
        print()
        print("Current folder:", Path(__file__).parent)
        input("Press Enter to exit...")
        return
    21
    # Show available files
    print("📁 Found these data files:")
    for i, file in enumerate(data_files, 1):
        print(f"   {i}. {file.name}")
    
    # Let user choose file
    if len(data_files) == 1:
        selected_file = data_files[0]
        print(f"\\n✅ Using: {selected_file.name}")
    else:
        print()
        try:
            choice = int(input("Enter file number: ")) - 1
            if 0 <= choice < len(data_files):
                selected_file = data_files[choice]
            else:
                print("❌ Invalid choice!")
                return
        except ValueError:
            print("❌ Please enter a number!")
            return
    
    # Get column name
    print()
    print("📝 What is the name of your text column?")
    print("   (The column containing the text responses you want to analyze)")
    print("   Common names: 'response', 'comment', 'feedback', 'text', 'answer'")
    
    column_name = input("Enter column name: ").strip()
    if not column_name:
        print("❌ Column name cannot be empty!")
        return
    
    # Set up environment
    print()
    print("🔧 Setting up environment...")
    python_exe = setup_environment()
    if not python_exe:
        return
    
    # Run analysis
    print("🔍 Running analysis...")
    print("   This may take a few minutes depending on your data size...")
    print()
    
    try:
        # Generate output filename
        output_filename = f"analysis_results_{selected_file.stem}.xlsx"
        
        cmd = [
            python_exe, "-m", "qualitative_analyzer", "analyze",
            str(selected_file),
            "--text-column", column_name,
            "--output", output_filename,
            "--report"
        ]
        
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print()
            print("🎉 Analysis completed successfully!")
            print("📊 Check the 'output' folder for your results:")
            print("   • Excel summary file")
            print("   • Word report document") 
            print("   • Theme charts")
        else:
            print()
            print("❌ Analysis failed. Common issues:")
            print("   • Wrong column name")
            print("   • Missing API credentials in .env file")
            print("   • Internet connection problems")
            
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
    
    print()
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()