"""
Main feature engineering module for flight delay prediction.

Combines temporal, route, and historical features.
"""

import logging
from typing import List

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
import yaml


class FlightFeatureEngineer:
    """Creates features for flight delay prediction."""

    def __init__(self, spark: SparkSession, config_path: str = "config/config.yaml"):
        """
        Initialize feature engineer.

        Args:
            spark: Active SparkSession
            config_path: Path to configuration file
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.feature_config = self.config.get("features", {})

    def create_temporal_features(self, df: DataFrame) -> DataFrame:
        """
        Create time-based features.

        Features created:
        - HOUR_OF_DAY: Hour extracted from scheduled departure time
        - TIME_OF_DAY: Category (Morning, Afternoon, Evening, Night)
        - IS_WEEKEND: Boolean for Saturday/Sunday
        - SEASON: Season based on month
        - QUARTER: Quarter of year

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with temporal features
        """
        self.logger.info("Creating temporal features")

        # Extract hour from departure time (format: HHMM)
        df = df.withColumn(
            "HOUR_OF_DAY", (F.col("CRS_DEP_TIME") / 100).cast("int")
        )

        # Time of day categories
        df = df.withColumn(
            "TIME_OF_DAY",
            F.when((F.col("HOUR_OF_DAY") >= 6) & (F.col("HOUR_OF_DAY") < 12), "Morning")
            .when((F.col("HOUR_OF_DAY") >= 12) & (F.col("HOUR_OF_DAY") < 17), "Afternoon")
            .when((F.col("HOUR_OF_DAY") >= 17) & (F.col("HOUR_OF_DAY") < 21), "Evening")
            .otherwise("Night"),
        )

        # Weekend indicator
        df = df.withColumn(
            "IS_WEEKEND", F.when(F.col("DAY_OF_WEEK").isin([6, 7]), 1).otherwise(0)
        )

        # Season
        df = df.withColumn(
            "SEASON",
            F.when(F.col("MONTH").isin([12, 1, 2]), "Winter")
            .when(F.col("MONTH").isin([3, 4, 5]), "Spring")
            .when(F.col("MONTH").isin([6, 7, 8]), "Summer")
            .otherwise("Fall"),
        )

        # Quarter
        df = df.withColumn("QUARTER", F.quarter(F.col("FL_DATE")))

        self.logger.info("Temporal features created")
        return df

    def create_holiday_features(self, df: DataFrame) -> DataFrame:
        """
        Create US federal holiday indicator.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with IS_HOLIDAY column
        """
        self.logger.info("Creating holiday features")

        # Simplified holiday detection (day before, day of, day after major holidays)
        # In production, use python holidays library for accurate dates
        holiday_periods = [
            (1, 1),  # New Year's
            (1, 18),  # MLK Day (3rd Monday of January, approximate)
            (2, 15),  # Presidents Day (3rd Monday of February, approximate)
            (5, 31),  # Memorial Day (last Monday of May, approximate)
            (7, 4),  # Independence Day
            (9, 6),  # Labor Day (1st Monday of September, approximate)
            (11, 11),  # Veterans Day
            (11, 25),  # Thanksgiving (4th Thursday of November, approximate)
            (12, 25),  # Christmas
        ]

        # Create holiday indicator
        is_holiday_expr = F.lit(0)
        for month, day in holiday_periods:
            # Check if within 1 day of holiday
            is_holiday_expr = F.when(
                (F.col("MONTH") == month) & (F.abs(F.col("DAY_OF_MONTH") - day) <= 1),
                1,
            ).otherwise(is_holiday_expr)

        df = df.withColumn("IS_HOLIDAY", is_holiday_expr)

        holiday_count = df.filter(F.col("IS_HOLIDAY") == 1).count()
        total_count = df.count()
        self.logger.info(
            f"Holiday flights: {holiday_count:,} ({holiday_count/total_count*100:.2f}%)"
        )

        return df

    def create_route_features(self, df: DataFrame) -> DataFrame:
        """
        Create route-based features.

        Features created:
        - ROUTE: Concatenation of ORIGIN-DEST
        - ROUTE_POPULARITY: Count of flights on this route
        - DISTANCE_CATEGORY: Short, Medium, Long, Very Long

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with route features
        """
        self.logger.info("Creating route features")

        # Create route identifier
        df = df.withColumn("ROUTE", F.concat_ws("-", F.col("ORIGIN"), F.col("DEST")))

        # Calculate route popularity
        route_counts = df.groupBy("ROUTE").agg(F.count("*").alias("ROUTE_POPULARITY"))
        df = df.join(route_counts, on="ROUTE", how="left")

        # Distance categories
        df = df.withColumn(
            "DISTANCE_CATEGORY",
            F.when(F.col("DISTANCE") < 500, "Short")
            .when((F.col("DISTANCE") >= 500) & (F.col("DISTANCE") < 1500), "Medium")
            .when((F.col("DISTANCE") >= 1500) & (F.col("DISTANCE") < 2500), "Long")
            .otherwise("Very Long"),
        )

        self.logger.info("Route features created")
        return df

    def create_historical_delay_features(self, df: DataFrame) -> DataFrame:
        """
        Create historical delay statistics by carrier and airport.

        Features created:
        - CARRIER_AVG_DELAY: Average delay for this carrier
        - ORIGIN_AVG_DELAY: Average delay for origin airport
        - DEST_AVG_DELAY: Average delay for destination airport

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with historical delay features
        """
        self.logger.info("Creating historical delay features")

        # Carrier average delay
        carrier_delays = df.groupBy("CARRIER").agg(
            F.avg("ARR_DELAY").alias("CARRIER_AVG_DELAY")
        )
        df = df.join(carrier_delays, on="CARRIER", how="left")

        # Origin airport average delay
        origin_delays = df.groupBy("ORIGIN").agg(
            F.avg("ARR_DELAY").alias("ORIGIN_AVG_DELAY")
        )
        df = df.join(origin_delays, on="ORIGIN", how="left")

        # Destination airport average delay
        dest_delays = df.groupBy("DEST").agg(
            F.avg("ARR_DELAY").alias("DEST_AVG_DELAY")
        )
        df = df.join(dest_delays, on="DEST", how="left")

        self.logger.info("Historical delay features created")
        return df

    def encode_categorical_features(
        self, df: DataFrame, categorical_cols: List[str]
    ) -> DataFrame:
        """
        Encode categorical features using StringIndexer.

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names

        Returns:
            DataFrame with encoded categorical features
        """
        self.logger.info(f"Encoding categorical features: {categorical_cols}")

        for col_name in categorical_cols:
            if col_name not in df.columns:
                self.logger.warning(f"Column {col_name} not found, skipping")
                continue

            indexer = StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_INDEX",
                handleInvalid="keep",
            )

            df = indexer.fit(df).transform(df)

        self.logger.info("Categorical encoding completed")
        return df

    def create_feature_vector(
        self, df: DataFrame, feature_cols: List[str], output_col: str = "features"
    ) -> DataFrame:
        """
        Assemble features into a single vector column.

        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            output_col: Name of output vector column

        Returns:
            DataFrame with assembled feature vector
        """
        self.logger.info(f"Assembling {len(feature_cols)} features into vector")

        # Filter to only existing columns
        existing_cols = [col for col in feature_cols if col in df.columns]

        if len(existing_cols) < len(feature_cols):
            missing = set(feature_cols) - set(existing_cols)
            self.logger.warning(f"Missing feature columns: {missing}")

        assembler = VectorAssembler(inputCols=existing_cols, outputCol=output_col)

        df = assembler.transform(df)

        self.logger.info(f"Feature vector created: {output_col}")
        return df

    def scale_features(
        self, df: DataFrame, input_col: str = "features", output_col: str = "scaled_features"
    ) -> DataFrame:
        """
        Scale features using StandardScaler.

        Args:
            df: Input DataFrame
            input_col: Input vector column
            output_col: Output scaled vector column

        Returns:
            DataFrame with scaled features
        """
        self.logger.info("Scaling features")

        scaler = StandardScaler(
            inputCol=input_col, outputCol=output_col, withMean=True, withStd=True
        )

        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)

        self.logger.info("Feature scaling completed")
        return df

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """
        Apply complete feature engineering pipeline.

        Args:
            df: Cleaned input DataFrame

        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("Starting feature engineering pipeline")

        # Temporal features
        df = self.create_temporal_features(df)
        df = self.create_holiday_features(df)

        # Route features
        df = self.create_route_features(df)

        # Historical features
        df = self.create_historical_delay_features(df)

        # Encode categorical features
        categorical_cols = self.config.get("preprocessing", {}).get(
            "categorical_features", []
        )
        categorical_cols += ["TIME_OF_DAY", "SEASON", "DISTANCE_CATEGORY"]
        df = self.encode_categorical_features(df, categorical_cols)

        self.logger.info("Feature engineering pipeline completed")
        return df


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Engineer features for flight delay prediction")
    parser.add_argument("--input", required=True, help="Input data path (cleaned data)")
    parser.add_argument("--output", required=True, help="Output path for features")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("FlightFeatureEngineering")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    try:
        # Load cleaned data
        logging.info(f"Loading cleaned data from: {args.input}")
        df = spark.read.parquet(args.input)

        # Initialize feature engineer
        feature_engineer = FlightFeatureEngineer(spark, args.config)

        # Engineer features
        df_features = feature_engineer.engineer_features(df)

        # Save features
        logging.info(f"Saving features to: {args.output}")
        df_features.write.mode("overwrite").partitionBy("MONTH").parquet(args.output)

        logging.info("Feature engineering completed successfully")

    except Exception as e:
        logging.error(f"Error during feature engineering: {str(e)}", exc_info=True)
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
