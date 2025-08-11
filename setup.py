"""Setup script for the Qualitative Text Analyzer."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AI-powered qualitative text analysis framework"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            lines = f.readlines()
        
        # Filter out comments and development dependencies
        requirements = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not any(dev in line.lower() for dev in ['pytest', 'black', 'isort', 'mypy']):
                requirements.append(line)
        
        return requirements
    return []

setup(
    name="qualitative-text-analyzer",
    version="1.0.0",
    author="Qualitative Analysis Team",
    author_email="dev@example.com",
    description="AI-powered qualitative text analysis framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qualitative-text-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "qualitative-analyzer=qualitative_analyzer.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "qualitative_analyzer": [
            "config/*.py",
            "*.md",
        ],
    },
    keywords="qualitative analysis, text analysis, AI, NLP, research, thematic analysis",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/qualitative-text-analyzer/issues",
        "Source": "https://github.com/yourusername/qualitative-text-analyzer",
        "Documentation": "https://github.com/yourusername/qualitative-text-analyzer/wiki",
    },
)