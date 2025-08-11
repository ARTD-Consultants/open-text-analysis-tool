11#!/usr/bin/env python3
"""Launcher script for Qualitative Text Analyzer with automatic virtual environment activation."""

import sys
import os
import subprocess
from pathlib import Path


def ensure_virtual_environment():
    """Ensure we're running in the virtual environment, activate if needed."""
    # Get the current script's directory (project root)
    script_dir = Path(__file__).parent
    venv_path = script_dir / "qualitative-analyzer-env"
    
    # Check if we're already in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Already in a virtual environment
        return True
    
    # Check if virtual environment exists
    if not venv_path.exists():
        print("❌ Virtual environment not found. Creating it now...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print("✅ Virtual environment created successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
    
    # Activate virtual environment and re-run the script
    python_executable = venv_path / "bin" / "python"
    if not python_executable.exists():
        python_executable = venv_path / "Scripts" / "python.exe"  # Windows
        
    if not python_executable.exists():
        print("❌ Virtual environment appears corrupted. Please recreate it.")
        return False
    
    # Check if dependencies are installed
    try:
        result = subprocess.run([str(python_executable), "-c", "import pandas"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("🔧 Installing dependencies...")
            subprocess.run([str(python_executable), "-m", "pip", "install", "-r", "requirements.txt"], 
                          check=True)
            print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    
    print(f"🔄 Running with virtual environment: {venv_path}")
    
    # Prepare the command
    if len(sys.argv) == 1:
        # If no arguments provided, show help instead of assuming default behavior
        cmd = [str(python_executable), "-m", "qualitative_analyzer", "--help"]
    else:
        cmd = [str(python_executable), "-m", "qualitative_analyzer"] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, cwd=script_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running in virtual environment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("🚀 Qualitative Text Analyzer Launcher")
    ensure_virtual_environment()