"""
Setup script for Flight Delay Prediction project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="flight-delay-prediction",
    version="1.0.0",
    description="PySpark-based flight delay prediction using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Flight Crew Team",
    author_email="team@example.com",
    url="https://github.com/yourusername/flight-delay-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "pylint>=3.0.3",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flight-ingest=preprocessing.data_ingestion:main",
            "flight-clean=preprocessing.data_cleaner:main",
            "flight-features=features.feature_engineering:main",
            "flight-train-lr=models.train_logistic_regression:main",
            "flight-train-rf=models.train_random_forest:main",
            "flight-evaluate=evaluation.model_evaluator:main",
            "flight-visualize=visualization.delay_analysis:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="machine-learning pyspark flight-delay prediction data-science",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/flight-delay-prediction/issues",
        "Source": "https://github.com/yourusername/flight-delay-prediction",
    },
)
