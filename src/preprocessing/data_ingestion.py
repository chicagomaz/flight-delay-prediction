"""
Flight data ingestion module.

Handles loading flight data from various sources (CSV, Parquet) using PySpark,
with support for schema validation and partitioned reads.
"""

import logging
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    DateType,
)
import yaml


class FlightDataIngestion:
    """Handles data ingestion for flight delay prediction."""

    def __init__(self, spark: SparkSession, config_path: str = "config/config.yaml"):
        """
        Initialize data ingestion handler.

        Args:
            spark: Active SparkSession
            config_path: Path to configuration file
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_flight_schema(self) -> StructType:
        """
        Define expected schema for flight data (BTS 2024 format).

        Returns:
            StructType with flight data schema
        """
        return StructType(
            [
                StructField("YEAR", IntegerType(), True),
                StructField("MONTH", IntegerType(), True),
                StructField("DAY_OF_MONTH", IntegerType(), True),
                StructField("DAY_OF_WEEK", IntegerType(), True),
                StructField("FL_DATE", StringType(), True),  # Will convert to date later
                StructField("OP_UNIQUE_CARRIER", StringType(), True),
                StructField("TAIL_NUM", StringType(), True),
                StructField("ORIGIN", StringType(), True),
                StructField("DEST", StringType(), True),
                StructField("CRS_DEP_TIME", IntegerType(), True),
                StructField("DEP_DELAY", DoubleType(), True),
                StructField("CRS_ARR_TIME", IntegerType(), True),
                StructField("ARR_DEL15", DoubleType(), True),  # Binary: 0 or 1 for >15 min delay
                StructField("CANCELLED", DoubleType(), True),
                StructField("CANCELLATION_CODE", StringType(), True),
                StructField("CRS_ELAPSED_TIME", DoubleType(), True),
                StructField("AIR_TIME", DoubleType(), True),
                StructField("DISTANCE", DoubleType(), True),
                StructField("CARRIER_DELAY", DoubleType(), True),
                StructField("WEATHER_DELAY", DoubleType(), True),
                StructField("NAS_DELAY", DoubleType(), True),
                StructField("SECURITY_DELAY", DoubleType(), True),
                StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True),
            ]
        )

    def load_csv(
        self, file_path: str, header: bool = True, infer_schema: bool = False
    ) -> DataFrame:
        """
        Load flight data from CSV file.

        Args:
            file_path: Path to CSV file
            header: Whether CSV has header row
            infer_schema: Whether to infer schema (slower) or use predefined

        Returns:
            Spark DataFrame with flight data
        """
        self.logger.info(f"Loading CSV data from: {file_path}")

        read_options = {"header": str(header).lower(), "mode": "PERMISSIVE"}

        if infer_schema:
            read_options["inferSchema"] = "true"
            df = self.spark.read.options(**read_options).csv(file_path)
        else:
            schema = self.get_flight_schema()
            df = self.spark.read.options(**read_options).schema(schema).csv(file_path)

        self.logger.info(f"Loaded {df.count():,} records")

        # Rename columns for pipeline compatibility
        if "OP_UNIQUE_CARRIER" in df.columns:
            df = df.withColumnRenamed("OP_UNIQUE_CARRIER", "CARRIER")
            self.logger.info("Renamed OP_UNIQUE_CARRIER to CARRIER")

        # If ARR_DEL15 exists, create ARR_DELAY column (compatibility)
        from pyspark.sql.functions import col
        if "ARR_DEL15" in df.columns and "ARR_DELAY" not in df.columns:
            # ARR_DEL15 is already binary (0/1), but create a placeholder ARR_DELAY
            # The cleaner will use ARR_DEL15 directly for IS_DELAYED
            df = df.withColumn("ARR_DELAY", col("ARR_DEL15") * 15.0)  # Approximate
            self.logger.info("Created ARR_DELAY from ARR_DEL15 (compatibility mode)")

        return df

    def load_parquet(self, file_path: str) -> DataFrame:
        """
        Load flight data from Parquet file(s).

        Args:
            file_path: Path to Parquet file or directory

        Returns:
            Spark DataFrame with flight data
        """
        self.logger.info(f"Loading Parquet data from: {file_path}")

        df = self.spark.read.parquet(file_path)

        self.logger.info(f"Loaded {df.count():,} records")
        return df

    def save_to_parquet(
        self,
        df: DataFrame,
        output_path: str,
        partition_by: Optional[list] = None,
        mode: str = "overwrite",
    ) -> None:
        """
        Save DataFrame to Parquet format for efficient storage and reading.

        Args:
            df: DataFrame to save
            output_path: Output path for Parquet files
            partition_by: Optional list of columns to partition by
            mode: Write mode (overwrite, append, etc.)
        """
        self.logger.info(f"Saving data to Parquet: {output_path}")

        writer = df.write.mode(mode)

        if partition_by:
            self.logger.info(f"Partitioning by: {partition_by}")
            writer = writer.partitionBy(*partition_by)

        writer.parquet(output_path)
        self.logger.info("Data saved successfully")

    def sample_data(self, df: DataFrame, fraction: float = 0.01, seed: int = 42) -> DataFrame:
        """
        Sample data for development and testing.

        Args:
            df: Input DataFrame
            fraction: Fraction of data to sample (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            Sampled DataFrame
        """
        self.logger.info(f"Sampling {fraction * 100}% of data")
        sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=seed)
        self.logger.info(f"Sampled {sampled_df.count():,} records")
        return sampled_df

    def get_data_summary(self, df: DataFrame) -> dict:
        """
        Get summary statistics for the dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "num_records": df.count(),
            "num_columns": len(df.columns),
            "columns": df.columns,
            "schema": str(df.schema),
        }

        self.logger.info("Data Summary:")
        self.logger.info(f"  Records: {summary['num_records']:,}")
        self.logger.info(f"  Columns: {summary['num_columns']}")

        return summary


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest flight delay data")
    parser.add_argument("--input", required=True, help="Input file path (CSV or Parquet)")
    parser.add_argument("--output", required=True, help="Output path for processed data")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Input format")
    parser.add_argument("--sample", type=float, default=1.0, help="Sample fraction (0.0-1.0)")
    parser.add_argument("--partition-by", nargs="+", help="Columns to partition by")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("FlightDataIngestion")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )

    try:
        # Initialize ingestion handler
        ingestion = FlightDataIngestion(spark, args.config)

        # Load data
        if args.format == "csv":
            df = ingestion.load_csv(args.input)
        else:
            df = ingestion.load_parquet(args.input)

        # Sample if requested
        if args.sample < 1.0:
            df = ingestion.sample_data(df, fraction=args.sample)

        # Get summary
        ingestion.get_data_summary(df)

        # Save to Parquet
        ingestion.save_to_parquet(df, args.output, partition_by=args.partition_by)

        logging.info("Data ingestion completed successfully")

    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}", exc_info=True)
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
