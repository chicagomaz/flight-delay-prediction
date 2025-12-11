"""
Schema validation module for flight data.

Ensures data conforms to expected schema and data types.
"""

import logging
from typing import List, Dict

from pyspark.sql import DataFrame
from pyspark.sql.types import DataType


class SchemaValidator:
    """Validates DataFrame schemas for flight data."""

    def __init__(self):
        """Initialize schema validator."""
        self.logger = logging.getLogger(__name__)

    def validate_required_columns(
        self, df: DataFrame, required_columns: List[str]
    ) -> bool:
        """
        Check if all required columns are present.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if all required columns present, False otherwise
        """
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False

        self.logger.info("All required columns present")
        return True

    def validate_data_types(
        self, df: DataFrame, expected_types: Dict[str, DataType]
    ) -> bool:
        """
        Validate that columns have expected data types.

        Args:
            df: DataFrame to validate
            expected_types: Dictionary mapping column names to expected types

        Returns:
            True if all types match, False otherwise
        """
        type_mismatches = []

        for col_name, expected_type in expected_types.items():
            if col_name not in df.columns:
                continue

            actual_type = df.schema[col_name].dataType

            if not isinstance(actual_type, type(expected_type)):
                type_mismatches.append(
                    {
                        "column": col_name,
                        "expected": expected_type,
                        "actual": actual_type,
                    }
                )

        if type_mismatches:
            self.logger.error("Data type mismatches found:")
            for mismatch in type_mismatches:
                self.logger.error(f"  {mismatch}")
            return False

        self.logger.info("All data types valid")
        return True

    def validate_value_ranges(
        self, df: DataFrame, range_constraints: Dict[str, tuple]
    ) -> bool:
        """
        Validate that numerical columns fall within expected ranges.

        Args:
            df: DataFrame to validate
            range_constraints: Dict mapping column names to (min, max) tuples

        Returns:
            True if all values within ranges, False otherwise
        """
        violations = []

        for col_name, (min_val, max_val) in range_constraints.items():
            if col_name not in df.columns:
                continue

            stats = df.agg({col_name: "min", col_name: "max"}).collect()[0]
            actual_min = stats[f"min({col_name})"]
            actual_max = stats[f"max({col_name})"]

            if actual_min < min_val or actual_max > max_val:
                violations.append(
                    {
                        "column": col_name,
                        "expected_range": (min_val, max_val),
                        "actual_range": (actual_min, actual_max),
                    }
                )

        if violations:
            self.logger.warning("Value range violations found:")
            for violation in violations:
                self.logger.warning(f"  {violation}")
            return False

        self.logger.info("All value ranges valid")
        return True

    def generate_schema_report(self, df: DataFrame) -> Dict:
        """
        Generate comprehensive schema report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with schema information
        """
        self.logger.info("Generating schema report")

        report = {
            "num_columns": len(df.columns),
            "columns": df.columns,
            "schema": {field.name: str(field.dataType) for field in df.schema.fields},
            "nullable_columns": [
                field.name for field in df.schema.fields if field.nullable
            ],
        }

        return report
