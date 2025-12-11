"""
Visualization module for flight delay analysis.

Creates charts and plots for delay patterns and model performance.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 300


class DelayVisualizer:
    """Creates visualizations for flight delay analysis."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize visualizer.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.viz_config = self.config.get("visualization", {})
        self.output_path = Path(self.config.get("output", {}).get("visualizations_path", "output/visualizations"))
        self.output_path.mkdir(parents=True, exist_ok=True)

    def plot_delay_by_carrier(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot delay rate by carrier.

        Args:
            df: Pandas DataFrame with CARRIER and IS_DELAYED columns
            save: Whether to save the plot
        """
        self.logger.info("Creating delay by carrier plot")

        # Calculate delay rate by carrier
        carrier_stats = df.groupby("CARRIER").agg(
            delay_rate=("IS_DELAYED", "mean"),
            total_flights=("IS_DELAYED", "count")
        ).reset_index()

        # Sort by delay rate
        carrier_stats = carrier_stats.sort_values("delay_rate", ascending=False)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        bars = ax.barh(carrier_stats["CARRIER"], carrier_stats["delay_rate"] * 100)

        # Color bars
        colors = self.viz_config.get("colors", {})
        for bar in bars:
            if bar.get_width() > 30:
                bar.set_color(colors.get("delayed", "#d62728"))
            else:
                bar.set_color(colors.get("on_time", "#2ca02c"))

        ax.set_xlabel("Delay Rate (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Carrier", fontsize=12, fontweight="bold")
        ax.set_title("Flight Delay Rate by Carrier", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save:
            output_file = self.output_path / "delay_by_carrier.png"
            plt.savefig(output_file, bbox_inches="tight")
            self.logger.info(f"Saved plot to: {output_file}")

        plt.close()

    def plot_delay_by_airport(self, df: pd.DataFrame, top_n: int = 20, save: bool = True) -> None:
        """
        Plot delay rate by airport (top N busiest).

        Args:
            df: Pandas DataFrame with ORIGIN and IS_DELAYED columns
            top_n: Number of top airports to show
            save: Whether to save the plot
        """
        self.logger.info(f"Creating delay by airport plot (top {top_n})")

        # Calculate delay rate by origin airport
        airport_stats = df.groupby("ORIGIN").agg(
            delay_rate=("IS_DELAYED", "mean"),
            total_flights=("IS_DELAYED", "count")
        ).reset_index()

        # Get top N busiest airports
        top_airports = airport_stats.nlargest(top_n, "total_flights")
        top_airports = top_airports.sort_values("delay_rate", ascending=False)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))

        bars = ax.barh(top_airports["ORIGIN"], top_airports["delay_rate"] * 100)

        # Color bars
        colors = self.viz_config.get("colors", {})
        for bar in bars:
            if bar.get_width() > 30:
                bar.set_color(colors.get("delayed", "#d62728"))
            else:
                bar.set_color(colors.get("on_time", "#2ca02c"))

        ax.set_xlabel("Delay Rate (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Airport Code", fontsize=12, fontweight="bold")
        ax.set_title(f"Flight Delay Rate by Airport (Top {top_n} Busiest)", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save:
            output_file = self.output_path / "delay_by_airport.png"
            plt.savefig(output_file, bbox_inches="tight")
            self.logger.info(f"Saved plot to: {output_file}")

        plt.close()

    def plot_delay_by_month(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot delay rate by month.

        Args:
            df: Pandas DataFrame with MONTH and IS_DELAYED columns
            save: Whether to save the plot
        """
        self.logger.info("Creating delay by month plot")

        # Calculate delay rate by month
        month_stats = df.groupby("MONTH").agg(
            delay_rate=("IS_DELAYED", "mean")
        ).reset_index()

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = self.viz_config.get("colors", {})
        ax.plot(month_stats["MONTH"], month_stats["delay_rate"] * 100,
               marker="o", linewidth=2, markersize=8, color=colors.get("primary", "#1f77b4"))

        ax.fill_between(month_stats["MONTH"], month_stats["delay_rate"] * 100,
                        alpha=0.3, color=colors.get("primary", "#1f77b4"))

        ax.set_xlabel("Month", fontsize=12, fontweight="bold")
        ax.set_ylabel("Delay Rate (%)", fontsize=12, fontweight="bold")
        ax.set_title("Flight Delay Rate by Month", fontsize=14, fontweight="bold")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            output_file = self.output_path / "delay_by_month.png"
            plt.savefig(output_file, bbox_inches="tight")
            self.logger.info(f"Saved plot to: {output_file}")

        plt.close()

    def plot_delay_by_day_of_week(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot delay rate by day of week.

        Args:
            df: Pandas DataFrame with DAY_OF_WEEK and IS_DELAYED columns
            save: Whether to save the plot
        """
        self.logger.info("Creating delay by day of week plot")

        # Calculate delay rate by day of week
        dow_stats = df.groupby("DAY_OF_WEEK").agg(
            delay_rate=("IS_DELAYED", "mean")
        ).reset_index()

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = self.viz_config.get("colors", {})
        bars = ax.bar(dow_stats["DAY_OF_WEEK"], dow_stats["delay_rate"] * 100,
                     color=colors.get("primary", "#1f77b4"), alpha=0.7, edgecolor="black")

        ax.set_xlabel("Day of Week", fontsize=12, fontweight="bold")
        ax.set_ylabel("Delay Rate (%)", fontsize=12, fontweight="bold")
        ax.set_title("Flight Delay Rate by Day of Week", fontsize=14, fontweight="bold")
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels(day_names, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save:
            output_file = self.output_path / "delay_by_day_of_week.png"
            plt.savefig(output_file, bbox_inches="tight")
            self.logger.info(f"Saved plot to: {output_file}")

        plt.close()

    def plot_delay_by_hour(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot delay rate by hour of day.

        Args:
            df: Pandas DataFrame with HOUR_OF_DAY and IS_DELAYED columns
            save: Whether to save the plot
        """
        self.logger.info("Creating delay by hour plot")

        # Calculate delay rate by hour
        hour_stats = df.groupby("HOUR_OF_DAY").agg(
            delay_rate=("IS_DELAYED", "mean")
        ).reset_index()

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))

        colors = self.viz_config.get("colors", {})
        ax.plot(hour_stats["HOUR_OF_DAY"], hour_stats["delay_rate"] * 100,
               marker="o", linewidth=2, markersize=6, color=colors.get("primary", "#1f77b4"))

        ax.fill_between(hour_stats["HOUR_OF_DAY"], hour_stats["delay_rate"] * 100,
                        alpha=0.3, color=colors.get("primary", "#1f77b4"))

        ax.set_xlabel("Hour of Day", fontsize=12, fontweight="bold")
        ax.set_ylabel("Delay Rate (%)", fontsize=12, fontweight="bold")
        ax.set_title("Flight Delay Rate by Hour of Day", fontsize=14, fontweight="bold")
        ax.set_xticks(range(0, 24))
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            output_file = self.output_path / "delay_by_hour.png"
            plt.savefig(output_file, bbox_inches="tight")
            self.logger.info(f"Saved plot to: {output_file}")

        plt.close()

    def plot_confusion_matrix(self, confusion_matrix: Dict, model_name: str = "Model", save: bool = True) -> None:
        """
        Plot confusion matrix.

        Args:
            confusion_matrix: Dictionary with TP, FP, TN, FN
            model_name: Name of the model
            save: Whether to save the plot
        """
        self.logger.info(f"Creating confusion matrix for {model_name}")

        # Create confusion matrix array
        cm = [
            [confusion_matrix["true_negatives"], confusion_matrix["false_positives"]],
            [confusion_matrix["false_negatives"], confusion_matrix["true_positives"]],
        ]

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted On-time", "Predicted Delayed"],
            yticklabels=["Actual On-time", "Actual Delayed"],
            ax=ax,
            cbar_kws={"label": "Count"},
        )

        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            output_file = self.output_path / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(output_file, bbox_inches="tight")
            self.logger.info(f"Saved plot to: {output_file}")

        plt.close()

    def plot_feature_importance(
        self, importance_dict: Dict, top_n: int = 15, save: bool = True
    ) -> None:
        """
        Plot feature importance from Random Forest.

        Args:
            importance_dict: Dictionary mapping feature names to importance scores
            top_n: Number of top features to show
            save: Whether to save the plot
        """
        self.logger.info(f"Creating feature importance plot (top {top_n})")

        # Convert to DataFrame and get top N
        importance_df = pd.DataFrame(
            list(importance_dict.items()), columns=["Feature", "Importance"]
        )

        importance_df = importance_df.nlargest(top_n, "Importance")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = self.viz_config.get("colors", {})
        bars = ax.barh(importance_df["Feature"], importance_df["Importance"],
                      color=colors.get("primary", "#1f77b4"), alpha=0.7, edgecolor="black")

        ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
        ax.set_title(f"Top {top_n} Most Important Features", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save:
            output_file = self.output_path / "feature_importance.png"
            plt.savefig(output_file, bbox_inches="tight")
            self.logger.info(f"Saved plot to: {output_file}")

        plt.close()

    def plot_model_comparison(self, model_metrics: List[Dict], save: bool = True) -> None:
        """
        Plot comparison of multiple models.

        Args:
            model_metrics: List of metric dictionaries from different models
            save: Whether to save the plot
        """
        self.logger.info(f"Creating model comparison plot for {len(model_metrics)} models")

        # Extract metrics for comparison
        models = [m["model_name"] for m in model_metrics]
        metrics_to_plot = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "auc_roc"]

        # Create DataFrame
        data = {metric: [m.get(metric, 0) for m in model_metrics] for metric in metrics_to_plot}
        df = pd.DataFrame(data, index=models)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        df.plot(kind="bar", ax=ax, width=0.8, edgecolor="black")

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save:
            output_file = self.output_path / "model_comparison.png"
            plt.savefig(output_file, bbox_inches="tight")
            self.logger.info(f"Saved plot to: {output_file}")

        plt.close()


def main():
    """Main function for standalone execution."""
    import argparse
    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser(description="Generate delay analysis visualizations")
    parser.add_argument("--data", required=True, help="Path to flight data (Parquet)")
    parser.add_argument("--output", default="output/visualizations", help="Output directory")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--sample-size", type=int, help="Sample size for faster processing")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize Spark
    spark = SparkSession.builder.appName("DelayVisualization").getOrCreate()

    try:
        # Load data
        logging.info(f"Loading data from: {args.data}")
        df_spark = spark.read.parquet(args.data)

        # Sample if specified
        if args.sample_size:
            logging.info(f"Sampling {args.sample_size} records")
            df_spark = df_spark.limit(args.sample_size)

        # Convert to Pandas for visualization
        logging.info("Converting to Pandas DataFrame")
        df_pandas = df_spark.select(
            "CARRIER", "ORIGIN", "DEST", "MONTH", "DAY_OF_WEEK",
            "HOUR_OF_DAY", "IS_DELAYED"
        ).toPandas()

        # Initialize visualizer
        visualizer = DelayVisualizer(args.config)

        # Generate all plots
        visualizer.plot_delay_by_carrier(df_pandas)
        visualizer.plot_delay_by_airport(df_pandas)
        visualizer.plot_delay_by_month(df_pandas)
        visualizer.plot_delay_by_day_of_week(df_pandas)
        visualizer.plot_delay_by_hour(df_pandas)

        logging.info("All visualizations generated successfully")

    except Exception as e:
        logging.error(f"Error during visualization: {str(e)}", exc_info=True)
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
