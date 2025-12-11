"""
Data preprocessing module for flight delay prediction.

This module contains utilities for:
- Data ingestion from CSV/Parquet sources
- Data cleaning and validation
- Missing value handling
- Schema enforcement
- Train/test split
"""

from .data_ingestion import FlightDataIngestion
from .data_cleaner import FlightDataCleaner
from .schema_validator import SchemaValidator

__all__ = [
    "FlightDataIngestion",
    "FlightDataCleaner",
    "SchemaValidator",
]
