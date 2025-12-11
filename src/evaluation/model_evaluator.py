"""
Comprehensive model evaluation module.

Provides detailed performance analysis and comparison across models.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import functions as F
import yaml


class ModelEvaluator:
    """Evaluates and compares machine learning models."""

    def __init__(self, spark: SparkSession, config_path: str = "config/config.yaml"):
        """
        Initialize evaluator.

        Args:
            spark: Active SparkSession
            config_path: Path to configuration file
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def evaluate_model(
        self, model: PipelineModel, test_df: DataFrame, model_name: str = "Model"
    ) -> Dict:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained model to evaluate
            test_df: Test DataFrame
            model_name: Name of the model for logging

        Returns:
            Dictionary with all evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name}")

        # Make predictions
        predictions = model.transform(test_df)

        # Binary classification metrics
        evaluator_roc = BinaryClassificationEvaluator(
            labelCol="IS_DELAYED", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )

        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="IS_DELAYED", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
        )

        auc_roc = evaluator_roc.evaluate(predictions)
        auc_pr = evaluator_pr.evaluate(predictions)

        # Multiclass metrics
        mc_evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="IS_DELAYED", predictionCol="prediction", metricName="accuracy"
        )

        mc_evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="IS_DELAYED", predictionCol="prediction", metricName="f1"
        )

        mc_evaluator_precision = MulticlassClassificationEvaluator(
            labelCol="IS_DELAYED",
            predictionCol="prediction",
            metricName="weightedPrecision",
        )

        mc_evaluator_recall = MulticlassClassificationEvaluator(
            labelCol="IS_DELAYED", predictionCol="prediction", metricName="weightedRecall"
        )

        accuracy = mc_evaluator_acc.evaluate(predictions)
        f1_weighted = mc_evaluator_f1.evaluate(predictions)
        precision_weighted = mc_evaluator_precision.evaluate(predictions)
        recall_weighted = mc_evaluator_recall.evaluate(predictions)

        # Confusion matrix
        confusion_matrix = self.compute_confusion_matrix(predictions)

        # Per-class metrics
        class_metrics = self.compute_per_class_metrics(predictions)

        # Combine all metrics
        metrics = {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "confusion_matrix": confusion_matrix,
            "class_0_metrics": class_metrics[0],
            "class_1_metrics": class_metrics[1],
        }

        # Log summary
        self.logger.info(f"{model_name} Performance:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Precision (weighted): {precision_weighted:.4f}")
        self.logger.info(f"  Recall (weighted): {recall_weighted:.4f}")
        self.logger.info(f"  F1 Score (weighted): {f1_weighted:.4f}")
        self.logger.info(f"  AUC-ROC: {auc_roc:.4f}")
        self.logger.info(f"  AUC-PR: {auc_pr:.4f}")

        return metrics

    def compute_confusion_matrix(self, predictions: DataFrame) -> Dict:
        """
        Compute confusion matrix.

        Args:
            predictions: DataFrame with predictions

        Returns:
            Dictionary with confusion matrix values
        """
        tp = predictions.filter("prediction = 1 AND IS_DELAYED = 1").count()
        fp = predictions.filter("prediction = 1 AND IS_DELAYED = 0").count()
        tn = predictions.filter("prediction = 0 AND IS_DELAYED = 0").count()
        fn = predictions.filter("prediction = 0 AND IS_DELAYED = 1").count()

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "total": tp + fp + tn + fn,
        }

    def compute_per_class_metrics(self, predictions: DataFrame) -> Dict:
        """
        Compute precision, recall, F1 for each class.

        Args:
            predictions: DataFrame with predictions

        Returns:
            Dictionary with per-class metrics
        """
        tp_0 = predictions.filter("prediction = 0 AND IS_DELAYED = 0").count()
        fp_0 = predictions.filter("prediction = 0 AND IS_DELAYED = 1").count()
        fn_0 = predictions.filter("prediction = 1 AND IS_DELAYED = 0").count()

        tp_1 = predictions.filter("prediction = 1 AND IS_DELAYED = 1").count()
        fp_1 = predictions.filter("prediction = 1 AND IS_DELAYED = 0").count()
        fn_1 = predictions.filter("prediction = 0 AND IS_DELAYED = 1").count()

        # Class 0 (On-time) metrics
        precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0
        recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0
        f1_0 = (
            2 * (precision_0 * recall_0) / (precision_0 + recall_0)
            if (precision_0 + recall_0) > 0
            else 0
        )

        # Class 1 (Delayed) metrics
        precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0
        recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0
        f1_1 = (
            2 * (precision_1 * recall_1) / (precision_1 + recall_1)
            if (precision_1 + recall_1) > 0
            else 0
        )

        return {
            0: {"precision": precision_0, "recall": recall_0, "f1": f1_0, "support": tp_0 + fn_0},
            1: {"precision": precision_1, "recall": recall_1, "f1": f1_1, "support": tp_1 + fn_1},
        }

    def compare_models(self, model_metrics: List[Dict]) -> Dict:
        """
        Compare multiple models and identify best performer.

        Args:
            model_metrics: List of metric dictionaries from different models

        Returns:
            Dictionary with comparison results
        """
        self.logger.info(f"Comparing {len(model_metrics)} models")

        # Find best model for each metric
        best_models = {}
        metrics_to_compare = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "auc_roc"]

        for metric in metrics_to_compare:
            best_model = max(model_metrics, key=lambda x: x.get(metric, 0))
            best_models[metric] = {
                "model": best_model["model_name"],
                "value": best_model.get(metric, 0),
            }

        # Log comparison
        self.logger.info("Best models by metric:")
        for metric, info in best_models.items():
            self.logger.info(f"  {metric}: {info['model']} ({info['value']:.4f})")

        return {
            "best_by_metric": best_models,
            "all_models": model_metrics,
        }

    def save_metrics(self, metrics: Dict, output_path: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            metrics: Metrics dictionary
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"Metrics saved to: {output_path}")


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--output", required=True, help="Output path for metrics JSON")
    parser.add_argument("--model-name", default="Model", help="Name of the model")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("ModelEvaluation")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    try:
        # Load model
        logging.info(f"Loading model from: {args.model}")
        model = PipelineModel.load(args.model)

        # Load test data
        logging.info(f"Loading test data from: {args.test_data}")
        test_df = spark.read.parquet(args.test_data)

        # Initialize evaluator
        evaluator = ModelEvaluator(spark, args.config)

        # Evaluate model
        metrics = evaluator.evaluate_model(model, test_df, args.model_name)

        # Save metrics
        evaluator.save_metrics(metrics, args.output)

        logging.info("Model evaluation completed successfully")

    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
