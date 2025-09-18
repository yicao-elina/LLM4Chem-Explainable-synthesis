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

# bash commands go here. Don't forget to load the necessary modules!
ml intel
# ml qe/7.2

ml
date
echo "This is using gemini-chem to extract doping experiments from scientific text."
# echo "Starting the process to fetch papers, convert PDFs to text, clean the text, and extract information."
# # python 1-fetch_papers_multi.py
# # python 2-pdf_to_txt.py
# # python 3-clean_txt.py
# echo "Extracting doping experiments from text files..."
# python 4-extract_info.py
echo "Running causal engine to analyze the extracted data..."
# python causal_engine.py
python causal_engine.py
# python evaluation_csv.py --engines "dag_engine:your_engine_module" --test_all_strategies --output_prefix "comprehensive_test"
date
