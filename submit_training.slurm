#!/bin/bash
#SBATCH --job-name=flight-delay-ml
#SBATCH --output=output/logs/slurm_%j.out
#SBATCH --error=output/logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --partition=open

# Flight Delay Prediction - ROAR Collab Training Job
# Submits the complete ML pipeline to Penn State ROAR Collab cluster

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   Flight Delay Prediction - ROAR Collab Training             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64GB"
echo "Start time: $(date)"
echo ""

# Using conda environment (no modules needed)
echo "📦 Using conda environment"
echo ""

# Set up Spark environment variables
export SPARK_HOME=$(python3 -c "import pyspark; import os; print(os.path.dirname(pyspark.__file__))")
export PYSPARK_PYTHON=$(which python3)
export PYSPARK_DRIVER_PYTHON=$(which python3)

# Optimize Spark for cluster
export SPARK_WORKER_MEMORY=60g
export SPARK_DRIVER_MEMORY=60g
export SPARK_EXECUTOR_MEMORY=60g

# Set up paths
PROJECT_DIR=$SLURM_SUBMIT_DIR
cd $PROJECT_DIR

echo "🗂️  Working directory: $PROJECT_DIR"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate ds410_f25
echo "✅ Conda environment activated: ds410_f25"

# Set JAVA_HOME from conda
export JAVA_HOME=$CONDA_PREFIX

echo ""
echo "🐍 Python: $(which python3)"
echo "📊 PySpark: $(python3 -c 'import pyspark; print(pyspark.__version__)')"
echo ""

# Get sample size (default 1.0 = 100%)
SAMPLE_SIZE="${1:-1.0}"
echo "📊 Training with sample size: $(echo "$SAMPLE_SIZE * 100" | bc)%"
echo ""

# Run the training pipeline
echo "🚀 Starting training pipeline..."
echo ""

./scripts/auto_train_after_download.sh $SAMPLE_SIZE

EXIT_CODE=$?

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║                    SUCCESS! ✅                                 ║"
else
    echo "║                    FAILED! ❌                                  ║"
fi
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "📁 Output locations:"
echo "   Models:         $PROJECT_DIR/output/models/"
echo "   Visualizations: $PROJECT_DIR/output/visualizations/"
echo "   Logs:           $PROJECT_DIR/output/logs/"
echo ""

exit $EXIT_CODE
