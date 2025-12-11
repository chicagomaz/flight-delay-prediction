# Flight Delay Prediction - ROAR Collab Guide

Complete guide for running the Flight Delay Prediction project on Penn State's ROAR Collab supercomputing cluster.

---

## 🚀 Quick Start

### 1. Connect to ROAR Collab

```bash
ssh YOUR_PSU_ID@submit.aci.ics.psu.edu
```

### 2. Upload Project to Cluster

**From your local machine:**
```bash
# Upload entire project (from your Windows WSL)
cd /mnt/c/Users/chica/Projects/PERSONALPROJECTS
scp -r flight-delay-predictionv2 YOUR_PSU_ID@submit.aci.ics.psu.edu:~/

# Or use rsync (faster for large files)
rsync -avz --progress flight-delay-predictionv2/ YOUR_PSU_ID@submit.aci.ics.psu.edu:~/flight-delay-predictionv2/
```

### 3. Setup Environment on Cluster

**On ROAR Collab:**
```bash
cd ~/flight-delay-predictionv2
chmod +x setup_roar.sh
./setup_roar.sh
```

**This will**:
- Load Python 3.11 and Java 11
- Create virtual environment
- Install all dependencies (PySpark, Pandas, etc.)
- Takes ~5-10 minutes

### 4. Submit Training Job

```bash
# Full dataset (31.3M records) - 3-4 hours
sbatch submit_training.slurm

# Or 10% sample for testing - ~30 minutes
sbatch submit_training.slurm 0.1

# Or 1% sample for quick test - ~5 minutes
sbatch submit_training.slurm 0.01
```

### 5. Monitor Job

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f output/logs/slurm_JOBID.out

# Check all your jobs
sacct -u $USER --format=JobID,JobName,State,Elapsed,CPUTime
```

---

## 📊 Understanding ROAR Collab Resources

### Allocated Resources

**In `submit_training.slurm`:**
```bash
#SBATCH --nodes=1              # 1 compute node
#SBATCH --ntasks=1             # 1 task
#SBATCH --cpus-per-task=16     # 16 CPU cores
#SBATCH --mem=64GB             # 64GB RAM
#SBATCH --time=4:00:00         # 4 hour time limit
#SBATCH --partition=open       # Open queue (free)
```

### Resource Recommendations

| Dataset Size | CPUs | Memory | Time Limit | Sample Size |
|--------------|------|--------|------------|-------------|
| 1% (313K)    | 4    | 16GB   | 0:30:00    | 0.01        |
| 10% (3.1M)   | 8    | 32GB   | 1:00:00    | 0.1         |
| 50% (15.7M)  | 12   | 48GB   | 2:00:00    | 0.5         |
| 100% (31.3M) | 16   | 64GB   | 4:00:00    | 1.0         |

### Adjusting Resources

Edit `submit_training.slurm`:

```bash
# For smaller job (10% sample):
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:00:00

# For larger job (more data):
#SBATCH --cpus-per-task=24
#SBATCH --mem=96GB
#SBATCH --time=8:00:00
```

---

## 📁 File Structure on ROAR Collab

```
~/flight-delay-predictionv2/
├── data/
│   ├── raw/
│   │   └── dot_flights_5years.csv     # Upload this file
│   ├── processed/                      # Created during pipeline
│   ├── processed_cleaned/              # Created during pipeline
│   └── parquet/                        # Created during pipeline
├── output/
│   ├── logs/
│   │   ├── slurm_JOBID.out           # SLURM output
│   │   ├── slurm_JOBID.err           # SLURM errors
│   │   └── training_*.log            # Training logs
│   ├── models/
│   │   ├── random_forest/
│   │   │   ├── metrics.json          # Model performance
│   │   │   └── feature_importance.json
│   │   └── logistic_regression/
│   └── visualizations/
│       ├── delay_by_carrier.png
│       ├── delay_by_airport.png
│       └── ...
├── submit_training.slurm              # Main job submission script
├── setup_roar.sh                      # Setup script
└── venv/                              # Python virtual environment
```

---

## 🔄 Common Workflows

### Workflow 1: Initial Setup and Test

```bash
# 1. Connect to cluster
ssh YOUR_PSU_ID@submit.aci.ics.psu.edu

# 2. Upload project
# (done from local machine)

# 3. Setup environment
cd ~/flight-delay-predictionv2
./setup_roar.sh

# 4. Test with 1% sample
sbatch submit_training.slurm 0.01

# 5. Monitor
watch -n 5 squeue -u $USER
tail -f output/logs/slurm_*.out
```

### Workflow 2: Full Training Run

```bash
# After testing, run full dataset
sbatch submit_training.slurm

# Check job progress
squeue -u $USER

# If job is running, tail the log
tail -f output/logs/slurm_JOBID.out

# After completion, check results
cat output/models/random_forest/metrics.json
ls output/visualizations/
```

### Workflow 3: Download Results

**From your local machine:**
```bash
# Download models and results
scp -r YOUR_PSU_ID@submit.aci.ics.psu.edu:~/flight-delay-predictionv2/output ./roar_results/

# Or specific files
scp YOUR_PSU_ID@submit.aci.ics.psu.edu:~/flight-delay-predictionv2/output/models/random_forest/metrics.json ./

# Download visualizations
scp YOUR_PSU_ID@submit.aci.ics.psu.edu:~/flight-delay-predictionv2/output/visualizations/*.png ./visualizations/
```

---

## 🛠️ Troubleshooting

### Job Not Starting?

```bash
# Check queue position
squeue -u $USER

# View detailed job info
scontrol show job JOBID

# Check partition availability
sinfo -p open
```

### Job Failed?

```bash
# View error log
cat output/logs/slurm_JOBID.err

# View training log
cat output/logs/training_*.log

# Check last 100 lines
tail -100 output/logs/slurm_JOBID.out
```

### Out of Memory?

**Reduce sample size or request more memory:**
```bash
# Edit submit_training.slurm
#SBATCH --mem=96GB  # Increase from 64GB

# Or use smaller sample
sbatch submit_training.slurm 0.5  # 50% instead of 100%
```

### Job Timeout?

**Increase time limit:**
```bash
# Edit submit_training.slurm
#SBATCH --time=8:00:00  # Increase from 4 hours
```

### Module Load Errors?

```bash
# Check available modules
module avail python
module avail java

# Try different versions
module load python/3.10.5
module load java/17.0.2
```

---

## 📈 Expected Performance

### Training Times (on ROAR Collab)

| Sample Size | Records    | CPUs | Time     | Output Size |
|-------------|------------|------|----------|-------------|
| 1%          | 313K       | 4    | ~5 min   | ~100MB      |
| 10%         | 3.1M       | 8    | ~30 min  | ~800MB      |
| 50%         | 15.7M      | 12   | ~2 hours | ~4GB        |
| 100%        | 31.3M      | 16   | ~4 hours | ~8GB        |

### Model Performance (Expected)

**With full 31.3M record dataset:**
- **Accuracy**: 85-90%
- **Precision**: 70-80%
- **Recall**: 65-75%
- **F1 Score**: 70-75%
- **AUC-ROC**: 0.75-0.85

---

## 🎓 For Your Project Report

### What to Include:

1. **Computational Resources Used:**
   ```
   Platform: Penn State ROAR Collab Supercomputing Cluster
   CPUs: 16 cores
   Memory: 64GB RAM
   Training Time: ~4 hours
   Dataset: 31.3 million flight records (3.2GB CSV)
   ```

2. **Processing Details:**
   ```
   Framework: Apache PySpark 3.5.0 (distributed computing)
   Data Cleaning: Removed 8% outliers, imputed missing values
   Features Engineered: 17 features (temporal, route, historical)
   Train/Val/Test Split: 70/15/15
   ```

3. **Model Results:**
   ```
   Models Trained: Logistic Regression, Random Forest, Decision Tree
   Best Model: Random Forest (100 trees, max depth 10)
   Performance: [Your actual metrics from output/models/]
   Feature Importance: [Top 5 from feature_importance.json]
   ```

---

## 🚨 Important Notes

### Storage Limits

- **Home directory**: 20GB quota (use for code and environment)
- **Work directory**: 512GB quota (use for data processing)
  ```bash
  # Move data to work directory
  mkdir -p $WORK/flight-delay-data
  cp data/raw/*.csv $WORK/flight-delay-data/
  ```

### Best Practices

1. **Always test with small sample first** (`0.01` or `0.1`)
2. **Monitor job progress** - don't just submit and forget
3. **Save results frequently** - download important outputs
4. **Clean up old files** - delete intermediate data after completion
5. **Check queue times** - weekend/evening jobs start faster

### Getting Help

- **ROAR Collab Documentation**: https://www.icds.psu.edu/roar-collab-user-guide/
- **Submit Help Ticket**: icds@psu.edu
- **Office Hours**: Check ICDS website for schedule

---

## 📞 Quick Reference Commands

```bash
# Job Management
sbatch submit_training.slurm        # Submit job
squeue -u $USER                     # Check status
scancel JOBID                       # Cancel job
scontrol show job JOBID             # Job details

# Resource Check
sinfo -p open                       # Partition info
squeue                              # All jobs in queue
sacct -u $USER                      # Your job history

# File Operations
du -sh ~/flight-delay-predictionv2  # Check disk usage
df -h $HOME                         # Home quota
df -h $WORK                         # Work quota

# Environment
module list                         # Loaded modules
source venv/bin/activate           # Activate Python env
which python3                       # Python location
```

---

## ✅ Submission Checklist

Before submitting your job:

- [ ] Uploaded data file to `data/raw/`
- [ ] Ran `./setup_roar.sh` successfully
- [ ] Tested with small sample (`0.01`)
- [ ] Reviewed resource allocation in `.slurm` file
- [ ] Checked disk space (`df -h $HOME`)
- [ ] Verified virtual environment works
- [ ] Ready to submit: `sbatch submit_training.slurm`

---

**Good luck with your project! 🚀**
