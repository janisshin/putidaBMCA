#!/bin/bash
#SBATCH --job-name=putidaBMCA        # Job name
#SBATCH --output=output.pgz           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=40           # Number of tasks per node
#SBATCH --mem=32G
#SBATCH --time=1:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=jshin1@uw.edu    # Email address for notifications

#Load necessary modules (if needed)
module load apptainer

#Your job commands go here
# apptainer build --force /gscratch/cheme/jshin1/putidabmca/putida-bmca.sif docker://janisshin/putidabmca:latest

srun apptainer run --bind /gscratch/cheme/jshin1/putidabmca:/putidabmca /gscratch/cheme/jshin1/putidabmca/putida-bmca.sif "--runName" "test" "--iter" "40000"
