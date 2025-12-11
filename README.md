# Airline Flight Delay Prediction with PySpark

**Course:** DS/CMPSC 410
**Semester:** Fall 2025
**Team:** Flight Crew

## Team Members
- Devin Charles (Data acquisition & preprocessing lead)
- Tanush Rai (PySpark pipeline lead)
- Mysha Rahman (Presentation lead)
- Milton La (Documentation lead)
- Timilehin Balogun (Visualization & results lead)
- Maximilian Menninga (ML lead)

## Project Overview

This project builds a scalable machine learning pipeline using PySpark to predict airline flight delays. We process large-scale U.S. Department of Transportation flight data (50GB+, 100M+ records) to classify flights as delayed (>15 minutes) or on-time.

### Key Features
- **Big Data Processing**: PySpark-based ETL pipeline for handling 50GB+ datasets
- **Machine Learning Models**: Logistic Regression, Decision Trees, and Random Forest classifiers
- **Scalability Testing**: Performance benchmarks across 1-4 node clusters
- **Comprehensive Visualization**: Delay pattern analysis by airline, airport, and time
- **Production-Ready**: Modular, documented code with unit tests

## Project Structure

```
flight-delay-predictionv2/
├── data/
│   ├── raw/              # Raw CSV data from DOT/Kaggle
│   ├── processed/        # Cleaned CSV data
│   └── parquet/          # Optimized Parquet format
├── src/
│   ├── preprocessing/    # Data cleaning and ingestion
│   ├── features/         # Feature engineering
│   ├── models/           # ML model training
│   ├── evaluation/       # Model evaluation and metrics
│   └── visualization/    # Charts and analysis plots
├── scripts/              # Spark-submit scripts for cluster
├── notebooks/            # Jupyter notebooks for exploration
├── config/               # Configuration files
├── tests/                # Unit tests
├── output/
│   ├── models/           # Trained model artifacts
│   ├── visualizations/   # Generated charts
│   └── benchmarks/       # Performance results
└── claudedocs/           # Project reports and documentation

```

## Quick Start

### Prerequisites
- Python 3.8+
- Apache Spark 3.x
- Java 8 or 11
- 16GB+ RAM recommended for local testing

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd flight-delay-predictionv2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Spark environment
export SPARK_HOME=/path/to/spark
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
```

### Running the Pipeline

#### 1. Data Preprocessing
```bash
# Local mode
spark-submit src/preprocessing/data_ingestion.py \
  --input data/raw/flights.csv \
  --output data/processed/

# Cluster mode
spark-submit --master yarn \
  --num-executors 4 \
  --executor-cores 4 \
  --executor-memory 8G \
  src/preprocessing/data_ingestion.py \
  --input /path/to/large/dataset.csv \
  --output /user/output/processed/
```

#### 2. Feature Engineering
```bash
spark-submit src/features/feature_engineering.py \
  --input data/processed/ \
  --output data/parquet/
```

#### 3. Model Training
```bash
# Logistic Regression
spark-submit src/models/train_logistic_regression.py \
  --input data/parquet/ \
  --output output/models/logistic_regression

# Random Forest
spark-submit src/models/train_random_forest.py \
  --input data/parquet/ \
  --output output/models/random_forest \
  --trees 100 \
  --depth 10
```

#### 4. Model Evaluation
```bash
spark-submit src/evaluation/evaluate_models.py \
  --model output/models/logistic_regression \
  --test-data data/parquet/test/
```

#### 5. Generate Visualizations
```bash
python src/visualization/delay_analysis.py \
  --data data/parquet/ \
  --output output/visualizations/
```

## Data Sources

### Primary Dataset
- **U.S. DOT On-Time Performance**: https://www.transtats.bts.gov/OT_Delay/
- Size: 100M+ records, 50GB+
- Format: CSV (converted to Parquet for efficiency)

### Development Dataset
- **Kaggle Flight Delay Dataset**: Used for local testing and validation
- https://www.kaggle.com/code/matviyamchislavskiy/flight-delay-prediction/input

## Machine Learning Pipeline

### Data Preprocessing
1. **Missing Value Handling**: Imputation strategies for key features
2. **Schema Validation**: Ensure consistent data types
3. **Data Partitioning**: Optimize for distributed processing

### Feature Engineering
- **Temporal Features**: Day of week, month, quarter, hour
- **Categorical Encoding**: Airline, origin/destination airports
- **Route Features**: Flight distance, route popularity
- **Holiday Indicators**: Major U.S. holidays
- **Historical Delay Features**: Airport/airline delay statistics

### Models Implemented

#### 1. Logistic Regression
- Fast baseline classifier
- Interpretable feature coefficients
- Good for class probability estimates

#### 2. Decision Tree
- Non-linear relationships
- Feature importance rankings
- Prone to overfitting (used for comparison)

#### 3. Random Forest (Primary Model)
- Ensemble of decision trees
- Handles class imbalance
- Robust feature importance
- Best overall performance

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Delayed flight prediction accuracy
- **Recall**: Percentage of actual delays caught
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model discrimination ability

## Performance Benchmarks

Performance tested across different cluster configurations:

| Nodes | Cores/Node | Records | Training Time | Accuracy |
|-------|------------|---------|---------------|----------|
| 1     | 4          | 1M      | TBD           | TBD      |
| 2     | 4          | 10M     | TBD           | TBD      |
| 4     | 4          | 50M     | TBD           | TBD      |

## Visualizations

Generated visualizations include:
1. **Delay Heatmaps**: By airport and airline
2. **Temporal Patterns**: Seasonal and daily delay trends
3. **Feature Importance**: Top predictors from tree models
4. **Model Comparison**: Performance across algorithms
5. **Scaling Analysis**: Runtime vs. cluster size

## Development Workflow

### Testing Locally
```bash
# Run unit tests
pytest tests/

# Test on sample data
spark-submit --master local[4] \
  src/models/train_logistic_regression.py \
  --input data/processed/sample_1000.csv \
  --output output/models/test_lr
```

### Cluster Deployment (ICDS)
```bash
# Submit to YARN cluster
./scripts/submit_training_job.sh \
  --nodes 4 \
  --model random_forest \
  --input /user/flight_data/full_dataset.parquet
```

## Challenges and Solutions

### Challenge 1: Data Size (50GB+)
**Solution**: Parquet format for 80% compression, partitioned by date for efficient reads

### Challenge 2: Class Imbalance
**Solution**: Stratified sampling, class weights in models, SMOTE for critical evaluation

### Challenge 3: PySpark Learning Curve
**Solution**: Incremental development on small samples, shared code repository, weekly sync meetings

### Challenge 4: Cluster Resource Limits
**Solution**: Data sampling for development, scheduled batch jobs, fallback to smaller test sets

### Challenge 5: Random Forest Computation Time
**Solution**: Parameter tuning on subsets, multi-node execution, early stopping criteria

## References

1. **U.S. Bureau of Transportation Statistics**: https://www.transtats.bts.gov/OT_Delay/
2. **Apache Spark Documentation**: https://spark.apache.org/docs/latest/
3. **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html
4. **Databricks Best Practices**: https://www.databricks.com/
5. **Kaggle Flight Delay Datasets**: https://www.kaggle.com/

## Contributing

Team members should:
1. Work on feature branches: `git checkout -b feature/your-feature`
2. Follow PEP 8 style guidelines
3. Add unit tests for new functions
4. Update documentation for API changes
5. Submit pull requests for review

## License

Academic project for DS/CMPSC 410, Fall 2025

## Contact

For questions or issues, contact team members via course communication channels.
