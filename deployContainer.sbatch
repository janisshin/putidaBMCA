#!/bin/bash
#SBATCH --job-name=putidaBMCA        # Job name
#SBATCH --output=output.pgz           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=40           # Number of tasks per node
#SBATCH --mem=32G
#SBATCH --time=30-0:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=jshin1@uw.edu    # Email address for notifications

#Load necessary modules (if needed)
module load apptainer

#Your job commands go here
apptainer build putida-bmca.sif docker://janisshin/putidabmca
srun apptainer run --bind /gscratch/cheme/jshin1/repos /gscratch/cheme/jshin1/putida-bmca.sif 
srun apptainer run /gscratch/cheme/jshin1/putida-bmca.sif 
# apptainer exec  "--runName test" "--iter 100"

#Optionally, you can include cleanup commands here (e.g., after the job finishes)
#For example:
#rm some_temp_file.txt

# srun apptainer run --bind /gscratch/cheme/dalba/repos/ECFERS /gscratch/cheme/dalba/ecfers_latest.sif ~/ipython_slurm/ecfers/launch_engine.sh