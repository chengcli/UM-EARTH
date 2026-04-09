#!/usr/bin/env bash

#SBATCH -J diagnostics              # job name
#SBATCH -A nrenno1                  # account (update this to your account)
#SBATCH -p standard                 # partition
#SBATCH -N 1                        # number of nodes
#SBATCH --mem=40G                   # request memory
#SBATCH --ntasks-per-node=1         # tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task (for multi-threading)
#SBATCH -t 02:00:00                 # wall time (2 hours)
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# Activate virtual environment
# Update this path to your Python virtual environment
source $HOME/pyenv/bin/activate

# Configuration
# Update these variables according to your setup
EXP_NAME=2026-02-08
LOCATION=ws-site1
EXP_DIR=${LOCATION}-${EXP_NAME}
INPUT_DIR=${PWD}/${EXP_DIR}
OUTPUT_DIR=${PWD}/${EXP_DIR}/plots
#TOPO_DIR=${PWD}/Topo/Data/Split/${LOCATION}
TOPO_DIR=${PWD}/input

# Number of threads for parallel processing
# Use the number of CPUs allocated
NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Change to the diagnostics directory
cd ${PWD}/diagnostics

echo "================================================"
echo "Starting Diagnostic Plotting Job"
echo "================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: $(hostname)"
echo "CPUs: ${NUM_THREADS}"
echo "Input Directory: ${INPUT_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Topography Directory: ${TOPO_DIR}"
echo "Location: ${LOCATION}"
echo "================================================"
echo ""

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the diagnostic plotting script with multi-threading
# Process all time indices and generate PDF with topography overlays
python3 generate_all_plots.py ${INPUT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --pdf ${OUTPUT_DIR}/frigate_diagnostic_plots.pdf \
    --all-times \
    --threads ${NUM_THREADS} \
    --topo-dir ${TOPO_DIR} \
    --location ${LOCATION} \
    --skip 2

# Check if the script succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Diagnostic plotting completed successfully!"
    echo "Output PDF: ${OUTPUT_DIR}/frigate_diagnostic_plots.pdf"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "ERROR: Diagnostic plotting failed!"
    echo "================================================"
    exit 1
fi

# Move log files to the output directory
mv ../${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out ${OUTPUT_DIR}/ 2>/dev/null || true
mv ../${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err ${OUTPUT_DIR}/ 2>/dev/null || true

echo "Log files moved to ${OUTPUT_DIR}"
echo ""
echo "Job completed at $(date)"
