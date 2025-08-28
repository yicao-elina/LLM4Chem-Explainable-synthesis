#!/bin/bash -l
#SBATCH --account=pclancy3
#SBATCH --export=ALL

#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --job-name="extract"
#SBATCH --output="slurm.%j"

#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

# module reset

source /data/apps/go.sh
source ~/.bashrc
conda activate gemini-chem

# export GOOGLE_API_KEY="your-api-key-here"
streamlit run material_design_app.py 