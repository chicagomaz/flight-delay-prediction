"""
Flight data cleaning module.

Handles missing values, outliers, and data quality issues.
"""

import logging
from typing import List, Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import yaml


class FlightDataCleaner:
    """Cleans and prepares flight data for analysis."""

    def __init__(self, spark: SparkSession, config_path: str = "config/config.yaml"):
        """
        Initialize data cleaner.

        Args:
            spark: Active SparkSession
            config_path: Path to configuration file
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.preprocessing_config = self.config.get("preprocessing", {})

    def remove_cancelled_flights(self, df: DataFrame) -> DataFrame:
        """
        Remove cancelled flights as they don't have meaningful delay data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame without cancelled flights
        """
        self.logger.info("Removing cancelled flights")
        initial_count = df.count()

        df_filtered = df.filter(F.col("CANCELLED") == 0)

        final_count = df_filtered.count()
        removed = initial_count - final_count
        self.logger.info(f"Removed {removed:,} cancelled flights ({removed/initial_count*100:.2f}%)")

        return df_filtered

    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Handle missing values using various strategies.

        Strategy:
        - Drop columns with >30% missing values
        - Drop rows with critical missing values (ORIGIN, DEST, CARRIER)
        - Impute numerical columns with median
        - Impute categorical columns with mode

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        self.logger.info("Handling missing values")

        # Calculate missing percentages
        total_rows = df.count()
        missing_stats = []

        for col_name in df.columns:
            missing_count = df.filter(F.col(col_name).isNull()).count()
            missing_pct = missing_count / total_rows
            missing_stats.append((col_name, missing_count, missing_pct))

            if missing_count > 0:
                self.logger.info(f"  {col_name}: {missing_pct*100:.2f}% missing")

        # Drop columns with too many missing values
        threshold = self.preprocessing_config.get("missing_threshold", 0.3)
        cols_to_drop = [col for col, _, pct in missing_stats if pct > threshold]

        if cols_to_drop:
            self.logger.info(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
            df = df.drop(*cols_to_drop)

        # Drop rows with missing critical columns
        critical_cols = ["ORIGIN", "DEST", "CARRIER", "FL_DATE"]
        existing_critical = [col for col in critical_cols if col in df.columns]

        if existing_critical:
            self.logger.info(f"Dropping rows with missing critical columns: {existing_critical}")
            df = df.dropna(subset=existing_critical)

        # Impute numerical columns with median
        numerical_cols = [
            field.name
            for field in df.schema.fields
            if isinstance(field.dataType, DoubleType) and field.name in df.columns
        ]

        for col_name in numerical_cols:
            if col_name != "ARR_DELAY":  # Don't impute target variable
                median_value = df.approxQuantile(col_name, [0.5], 0.01)[0]
                df = df.fillna({col_name: median_value})
                self.logger.info(f"  Imputed {col_name} with median: {median_value:.2f}")

        return df

    def create_target_variable(self, df: DataFrame) -> DataFrame:
        """
        Create binary target variable for classification.

        Flight is considered delayed if ARR_DELAY > threshold (default 15 minutes).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with IS_DELAYED column
        """
        delay_threshold = self.preprocessing_config.get("delay_threshold_minutes", 15)
        self.logger.info(f"Creating target variable (delay threshold: {delay_threshold} minutes)")

        # If ARR_DEL15 exists (BTS 2024 format), use it directly
        if "ARR_DEL15" in df.columns:
            self.logger.info("Using ARR_DEL15 as IS_DELAYED (already binary)")
            df = df.withColumn("IS_DELAYED", F.col("ARR_DEL15").cast("integer"))
        else:
            # Otherwise calculate from ARR_DELAY
            df = df.withColumn(
                "IS_DELAYED", F.when(F.col("ARR_DELAY") > delay_threshold, 1).otherwise(0)
            )

        # Show class distribution
        delay_counts = df.groupBy("IS_DELAYED").count().collect()
        for row in delay_counts:
            label = "Delayed" if row["IS_DELAYED"] == 1 else "On-time"
            count = row["count"]
            pct = count / df.count() * 100
            self.logger.info(f"  {label}: {count:,} ({pct:.2f}%)")

        return df

    def remove_outliers(self, df: DataFrame, columns: List[str] = None) -> DataFrame:
        """
        Remove outliers using IQR method.

        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers (default: numerical columns)

        Returns:
            DataFrame with outliers removed
        """
        if columns is None:
            columns = ["DEP_DELAY", "DISTANCE", "CRS_ELAPSED_TIME"]  # Removed ARR_DELAY (binary-derived)

        self.logger.info(f"Removing outliers from columns: {columns}")
        initial_count = df.count()

        for col_name in columns:
            if col_name not in df.columns:
                continue

            # Calculate Q1, Q3, and IQR
            quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1

            # Define outlier bounds
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr

            # Filter outliers
            df = df.filter(
                (F.col(col_name) >= lower_bound) & (F.col(col_name) <= upper_bound)
            )

            self.logger.info(
                f"  {col_name}: bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
            )

        final_count = df.count()
        removed = initial_count - final_count
        self.logger.info(f"Removed {removed:,} outlier records ({removed/initial_count*100:.2f}%)")

        return df

    def validate_data_quality(self, df: DataFrame) -> Dict:
        """
        Validate data quality and return quality metrics.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with quality metrics
        """
        self.logger.info("Validating data quality")

        metrics = {
            "total_records": df.count(),
            "total_columns": len(df.columns),
        }

        # Check for duplicates
        duplicate_count = df.count() - df.dropDuplicates().count()
        metrics["duplicate_records"] = duplicate_count

        # Check for negative delays (valid but unusual)
        if "ARR_DELAY" in df.columns:
            negative_delays = df.filter(F.col("ARR_DELAY") < -60).count()
            metrics["negative_delays_over_60min"] = negative_delays

        # Check for invalid airports (not 3-letter IATA codes)
        if "ORIGIN" in df.columns:
            invalid_origins = df.filter(F.length(F.col("ORIGIN")) != 3).count()
            metrics["invalid_origin_codes"] = invalid_origins

        if "DEST" in df.columns:
            invalid_dests = df.filter(F.length(F.col("DEST")) != 3).count()
            metrics["invalid_dest_codes"] = invalid_dests

        # Log quality metrics
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:,}")

        return metrics

    def clean_data(self, df: DataFrame) -> DataFrame:
        """
        Apply full cleaning pipeline.

        Args:
            df: Raw input DataFrame

        Returns:
            Cleaned DataFrame ready for feature engineering
        """
        self.logger.info("Starting data cleaning pipeline")

        # Step 1: Remove cancelled flights
        df = self.remove_cancelled_flights(df)

        # Step 2: Handle missing values
        df = self.handle_missing_values(df)

        # Step 3: Create target variable
        df = self.create_target_variable(df)

        # Step 4: Remove extreme outliers
        df = self.remove_outliers(df)

        # Step 5: Validate quality
        quality_metrics = self.validate_data_quality(df)

        self.logger.info("Data cleaning pipeline completed")
        self.logger.info(f"Final record count: {df.count():,}")

        return df


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean flight delay data")
    parser.add_argument("--input", required=True, help="Input data path")
    parser.add_argument("--output", required=True, help="Output path for cleaned data")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("FlightDataCleaning")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    try:
        # Load data
        logging.info(f"Loading data from: {args.input}")
        df = spark.read.parquet(args.input)

        # Initialize cleaner and clean data
        cleaner = FlightDataCleaner(spark, args.config)
        df_cleaned = cleaner.clean_data(df)

        # Save cleaned data
        logging.info(f"Saving cleaned data to: {args.output}")
        df_cleaned.write.mode("overwrite").parquet(args.output)

        logging.info("Data cleaning completed successfully")

    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}", exc_info=True)
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
