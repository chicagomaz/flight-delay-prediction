#!/bin/bash
# Automated training pipeline after data download

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Flight Delay Prediction - Automated Training Pipeline    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
SAMPLE_SIZE="${1:-1.0}"  # Default: use all data (1.0 = 100%)
DATA_FILE=""

# Find the data file
echo "🔍 Step 1: Locating data file..."
if [ -f "data/raw/dot_flights.csv" ]; then
    DATA_FILE="data/raw/dot_flights.csv"
    echo "   ✅ Found: data/raw/dot_flights.csv"
elif [ -f "data/raw/kaggle_flights.csv" ]; then
    DATA_FILE="data/raw/kaggle_flights.csv"
    echo "   ✅ Found: data/raw/kaggle_flights.csv"
else
    # Find any CSV in data/raw
    CSV_FILE=$(find data/raw -name "*.csv" -type f | head -1)
    if [ -n "$CSV_FILE" ]; then
        DATA_FILE="$CSV_FILE"
        echo "   ✅ Found: $DATA_FILE"
    else
        echo "   ❌ Error: No CSV files found in data/raw/"
        echo ""
        echo "   Please download data first!"
        echo "   See: scripts/download_instructions.md"
        echo ""
        exit 1
    fi
fi

# Get file info
FILE_SIZE=$(du -h "$DATA_FILE" | cut -f1)
LINE_COUNT=$(wc -l < "$DATA_FILE")
RECORD_COUNT=$((LINE_COUNT - 1))

echo ""
echo "📊 Dataset Information:"
echo "   File: $DATA_FILE"
echo "   Size: $FILE_SIZE"
echo "   Records: $(printf "%'d" $RECORD_COUNT)"
echo "   Sample: $(echo "$SAMPLE_SIZE * 100" | bc)%"
echo ""

# Ask for confirmation
if [ "$SAMPLE_SIZE" = "1.0" ] && [ $RECORD_COUNT -gt 1000000 ]; then
    echo "⚠️  Large dataset detected (>1M records)"
    echo "   This may take significant time and resources."
    echo ""
    echo "   Recommended for first run: Use sample mode"
    echo "   Example: ./scripts/auto_train_after_download.sh 0.1  (10% sample)"
    echo ""
    read -p "   Continue with full dataset? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Cancelled. Try with: ./scripts/auto_train_after_download.sh 0.1"
        exit 0
    fi
fi

echo "🚀 Starting pipeline..."
echo ""

# Create output directories
mkdir -p data/processed
mkdir -p data/parquet
mkdir -p output/models
mkdir -p output/visualizations
mkdir -p output/logs

# Log file
LOG_FILE="output/logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Logging to: $LOG_FILE"
echo ""

# Function to log and display
log_step() {
    echo "$1" | tee -a "$LOG_FILE"
}

# ============================================
# STEP 1: Data Ingestion
# ============================================
log_step "╔═══════════════════════════════════════════════════════════════╗"
log_step "║ STEP 1/6: Data Ingestion                                     ║"
log_step "╚═══════════════════════════════════════════════════════════════╝"

spark-submit src/preprocessing/data_ingestion.py \
    --input "$DATA_FILE" \
    --output data/processed/ \
    --format csv \
    --sample "$SAMPLE_SIZE" \
    --config config/config.yaml 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
log_step "✅ Data ingestion complete!"
echo "" | tee -a "$LOG_FILE"

# ============================================
# STEP 2: Data Cleaning
# ============================================
log_step "╔═══════════════════════════════════════════════════════════════╗"
log_step "║ STEP 2/6: Data Cleaning                                      ║"
log_step "╚═══════════════════════════════════════════════════════════════╝"

spark-submit src/preprocessing/data_cleaner.py \
    --input data/processed/ \
    --output data/processed_cleaned/ \
    --config config/config.yaml 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
log_step "✅ Data cleaning complete!"
echo "" | tee -a "$LOG_FILE"

# ============================================
# STEP 3: Feature Engineering
# ============================================
log_step "╔═══════════════════════════════════════════════════════════════╗"
log_step "║ STEP 3/6: Feature Engineering                                ║"
log_step "╚═══════════════════════════════════════════════════════════════╝"

spark-submit src/features/feature_engineering.py \
    --input data/processed_cleaned/ \
    --output data/parquet/ \
    --config config/config.yaml 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
log_step "✅ Feature engineering complete!"
echo "" | tee -a "$LOG_FILE"

# ============================================
# STEP 4: Train Logistic Regression
# ============================================
log_step "╔═══════════════════════════════════════════════════════════════╗"
log_step "║ STEP 4/6: Training Logistic Regression Model                 ║"
log_step "╚═══════════════════════════════════════════════════════════════╝"

spark-submit src/models/train_logistic_regression.py \
    --input data/parquet/ \
    --output output/models/logistic_regression \
    --config config/config.yaml \
    --metrics-output output/models/logistic_regression/metrics.json 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
log_step "✅ Logistic Regression training complete!"
echo "" | tee -a "$LOG_FILE"

# Show LR results
if [ -f "output/models/logistic_regression/metrics.json" ]; then
    log_step "📊 Logistic Regression Results:"
    cat output/models/logistic_regression/metrics.json | python3 -m json.tool | grep -E "accuracy|precision|recall|f1_score" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# ============================================
# STEP 5: Train Random Forest
# ============================================
log_step "╔═══════════════════════════════════════════════════════════════╗"
log_step "║ STEP 5/6: Training Random Forest Model (This may take time)  ║"
log_step "╚═══════════════════════════════════════════════════════════════╝"

# Adjust trees based on sample size
if (( $(echo "$SAMPLE_SIZE < 0.2" | bc -l) )); then
    TREES=50
    log_step "   Using 50 trees (small sample)"
else
    TREES=100
    log_step "   Using 100 trees (default)"
fi

spark-submit src/models/train_random_forest.py \
    --input data/parquet/ \
    --output output/models/random_forest \
    --config config/config.yaml \
    --trees $TREES \
    --metrics-output output/models/random_forest/metrics.json \
    --importance-output output/models/random_forest/feature_importance.json 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
log_step "✅ Random Forest training complete!"
echo "" | tee -a "$LOG_FILE"

# Show RF results
if [ -f "output/models/random_forest/metrics.json" ]; then
    log_step "📊 Random Forest Results:"
    cat output/models/random_forest/metrics.json | python3 -m json.tool | grep -E "accuracy|precision|recall|f1_score|auc" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# Show top features
if [ -f "output/models/random_forest/feature_importance.json" ]; then
    log_step "🎯 Top 5 Most Important Features:"
    python3 -c "
import json
with open('output/models/random_forest/feature_importance.json', 'r') as f:
    data = json.load(f)
    sorted_features = sorted(data.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (feature, importance) in enumerate(sorted_features, 1):
        print(f'   {i}. {feature}: {importance:.4f}')
" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# ============================================
# STEP 6: Generate Visualizations
# ============================================
log_step "╔═══════════════════════════════════════════════════════════════╗"
log_step "║ STEP 6/6: Generating Visualizations                          ║"
log_step "╚═══════════════════════════════════════════════════════════════╝"

# Determine sample size for visualization
VIZ_SAMPLE=100000
if [ $RECORD_COUNT -lt 100000 ]; then
    VIZ_SAMPLE=$RECORD_COUNT
fi

python3 src/visualization/delay_analysis.py \
    --data data/parquet/ \
    --output output/visualizations/ \
    --config config/config.yaml \
    --sample-size $VIZ_SAMPLE 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
log_step "✅ Visualizations complete!"
echo "" | tee -a "$LOG_FILE"

# ============================================
# FINAL SUMMARY
# ============================================
log_step "╔═══════════════════════════════════════════════════════════════╗"
log_step "║                    PIPELINE COMPLETED! 🎉                      ║"
log_step "╚═══════════════════════════════════════════════════════════════╝"
echo "" | tee -a "$LOG_FILE"

log_step "📊 Results Summary:"
log_step "───────────────────────────────────────────────────────────────"
echo "" | tee -a "$LOG_FILE"

# Model comparison
log_step "🤖 Model Performance:"
if [ -f "output/models/logistic_regression/metrics.json" ] && [ -f "output/models/random_forest/metrics.json" ]; then
    python3 -c "
import json

with open('output/models/logistic_regression/metrics.json', 'r') as f:
    lr = json.load(f)
with open('output/models/random_forest/metrics.json', 'r') as f:
    rf = json.load(f)

print('   ┌─────────────────────┬──────────────┬──────────────┐')
print('   │ Metric              │ Log. Reg.    │ Random Forest│')
print('   ├─────────────────────┼──────────────┼──────────────┤')
print(f\"   │ Accuracy            │ {lr.get('accuracy', 0):.4f}       │ {rf.get('accuracy', 0):.4f}       │\")
print(f\"   │ Precision           │ {lr.get('precision', 0):.4f}       │ {rf.get('precision_weighted', 0):.4f}       │\")
print(f\"   │ Recall              │ {lr.get('recall', 0):.4f}       │ {rf.get('recall_weighted', 0):.4f}       │\")
print(f\"   │ F1-Score            │ {lr.get('f1_score', 0):.4f}       │ {rf.get('f1_weighted', 0):.4f}       │\")
print(f\"   │ AUC-ROC             │ {lr.get('auc_roc', 0):.4f}       │ {rf.get('auc_roc', 0):.4f}       │\")
print('   └─────────────────────┴──────────────┴──────────────┘')
" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
log_step "📁 Output Locations:"
log_step "   Models:         output/models/"
log_step "   Visualizations: output/visualizations/"
log_step "   Metrics:        output/models/*/metrics.json"
log_step "   Log file:       $LOG_FILE"
echo "" | tee -a "$LOG_FILE"

log_step "📈 Generated Visualizations:"
for viz in output/visualizations/*.png; do
    if [ -f "$viz" ]; then
        log_step "   ✅ $(basename "$viz")"
    fi
done

echo "" | tee -a "$LOG_FILE"
log_step "🎯 Next Steps:"
log_step "   1. View results:        cat output/models/random_forest/metrics.json"
log_step "   2. View visualizations: open output/visualizations/"
log_step "   3. Check full log:      cat $LOG_FILE"
log_step "   4. Create presentation: See output/for_presentation/"
echo "" | tee -a "$LOG_FILE"

log_step "✨ All done! Your models are trained and ready to analyze."
echo ""
