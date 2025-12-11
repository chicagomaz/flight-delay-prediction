"""
Decision Tree model training for flight delay prediction.

Used for comparison with other models.
"""

import logging
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import yaml


class DecisionTreeTrainer:
    """Trains Decision Tree model for flight delay prediction."""

    def __init__(self, spark: SparkSession, config_path: str = "config/config.yaml"):
        """
        Initialize trainer.

        Args:
            spark: Active SparkSession
            config_path: Path to configuration file
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config.get("models", {}).get("decision_tree", {})

    def prepare_features(self, df: DataFrame, feature_cols: list) -> DataFrame:
        """
        Prepare feature vector for training.

        Args:
            df: Input DataFrame with engineered features
            feature_cols: List of feature column names

        Returns:
            DataFrame with assembled features
        """
        self.logger.info(f"Preparing {len(feature_cols)} features")

        # Filter to existing columns
        existing_cols = [col for col in feature_cols if col in df.columns]

        if len(existing_cols) < len(feature_cols):
            missing = set(feature_cols) - set(existing_cols)
            self.logger.warning(f"Missing columns: {missing}")

        assembler = VectorAssembler(
            inputCols=existing_cols, outputCol="features", handleInvalid="skip"
        )

        df = assembler.transform(df)
        return df

    def split_data(self, df: DataFrame) -> tuple:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        preprocessing_config = self.config.get("preprocessing", {})
        train_ratio = preprocessing_config.get("train_ratio", 0.7)
        val_ratio = preprocessing_config.get("validation_ratio", 0.15)
        test_ratio = preprocessing_config.get("test_ratio", 0.15)
        seed = preprocessing_config.get("random_seed", 42)

        self.logger.info(f"Splitting data: train={train_ratio}, val={val_ratio}, test={test_ratio}")

        train_df, temp_df = df.randomSplit([train_ratio, val_ratio + test_ratio], seed=seed)
        val_df, test_df = temp_df.randomSplit(
            [val_ratio / (val_ratio + test_ratio), test_ratio / (val_ratio + test_ratio)],
            seed=seed,
        )

        self.logger.info(f"Train set: {train_df.count():,} records")
        self.logger.info(f"Validation set: {val_df.count():,} records")
        self.logger.info(f"Test set: {test_df.count():,} records")

        return train_df, val_df, test_df

    def train_model(self, train_df: DataFrame) -> Pipeline:
        """
        Train Decision Tree model.

        Args:
            train_df: Training DataFrame

        Returns:
            Trained Pipeline model
        """
        self.logger.info("Training Decision Tree model")

        # Get model hyperparameters from config
        max_depth = self.model_config.get("max_depth", 10)
        max_bins = self.model_config.get("max_bins", 32)
        min_instances = self.model_config.get("min_instances_per_node", 1)
        impurity = self.model_config.get("impurity", "gini")

        self.logger.info(f"Model configuration:")
        self.logger.info(f"  Max depth: {max_depth}")
        self.logger.info(f"  Max bins: {max_bins}")
        self.logger.info(f"  Impurity: {impurity}")

        # Create Decision Tree classifier
        dt = DecisionTreeClassifier(
            featuresCol="features",
            labelCol="IS_DELAYED",
            maxDepth=max_depth,
            maxBins=max_bins,
            minInstancesPerNode=min_instances,
            impurity=impurity,
        )

        # Create pipeline
        pipeline = Pipeline(stages=[dt])

        # Train model
        model = pipeline.fit(train_df)

        self.logger.info("Model training completed")
        return model

    def evaluate_model(self, model: Pipeline, test_df: DataFrame) -> dict:
        """
        Evaluate trained model on test data.

        Args:
            model: Trained model
            test_df: Test DataFrame

        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("Evaluating model on test data")

        # Make predictions
        predictions = model.transform(test_df)

        # Calculate metrics
        evaluator_roc = BinaryClassificationEvaluator(
            labelCol="IS_DELAYED", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )

        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="IS_DELAYED", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
        )

        auc_roc = evaluator_roc.evaluate(predictions)
        auc_pr = evaluator_pr.evaluate(predictions)

        # Confusion matrix
        tp = predictions.filter("prediction = 1 AND IS_DELAYED = 1").count()
        fp = predictions.filter("prediction = 1 AND IS_DELAYED = 0").count()
        tn = predictions.filter("prediction = 0 AND IS_DELAYED = 0").count()
        fn = predictions.filter("prediction = 0 AND IS_DELAYED = 1").count()

        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

        # Log metrics
        self.logger.info("Evaluation Results:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric_name}: {value:.4f}")
            else:
                self.logger.info(f"  {metric_name}: {value:,}")

        return metrics


def main():
    """Main function for standalone execution."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train Decision Tree model")
    parser.add_argument("--input", required=True, help="Input feature data path")
    parser.add_argument("--output", required=True, help="Output path for trained model")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--metrics-output", help="Output path for metrics JSON")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("DecisionTreeTraining")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    try:
        # Load feature data
        logging.info(f"Loading feature data from: {args.input}")
        df = spark.read.parquet(args.input)

        # Initialize trainer
        trainer = DecisionTreeTrainer(spark, args.config)

        # Define feature columns
        feature_cols = [
            "CARRIER_INDEX",
            "ORIGIN_INDEX",
            "DEST_INDEX",
            "DAY_OF_WEEK",
            "MONTH",
            "HOUR_OF_DAY",
            "IS_WEEKEND",
            "IS_HOLIDAY",
            "TIME_OF_DAY_INDEX",
            "SEASON_INDEX",
            "DISTANCE_CATEGORY_INDEX",
            "DISTANCE",
            "CRS_ELAPSED_TIME",
            "ROUTE_POPULARITY",
            "CARRIER_AVG_DELAY",
            "ORIGIN_AVG_DELAY",
            "DEST_AVG_DELAY",
        ]

        # Prepare features
        df = trainer.prepare_features(df, feature_cols)

        # Split data
        train_df, val_df, test_df = trainer.split_data(df)

        # Train model
        model = trainer.train_model(train_df)

        # Evaluate model
        metrics = trainer.evaluate_model(model, test_df)

        # Save model
        logging.info(f"Saving model to: {args.output}")
        model.write().overwrite().save(args.output)

        # Save metrics if output path provided
        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logging.info(f"Metrics saved to: {args.metrics_output}")

        logging.info("Decision Tree training completed successfully")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
