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
ml qe/7.2

ml
date
echo "This is using gemini-chem to evaluate doping experiments from scientific text."
# echo "Starting the process to fetch papers, convert PDFs to text, clean the text, and extract information."
# # python 1-fetch_papers_multi.py
# # python 2-pdf_to_txt.py
# # python 3-clean_txt.py
# echo "Extracting doping experiments from text files..."
# python 4-extract_info.py
echo "Running evaluation script to analyze the engine results..."
# python evaluation_test.py
### 08/28/2025
# python evaluation_test.py /data/pclancy3/yi/26ICLR/outputs/combined_doping_data.json

### 08/29/2025
# python evaluation0829.py /data/pclancy3/yi/26ICLR/outputs/combined_doping_data.json
# python evaluation_plot.py

### Multi-engine
# python evaluation_multi.py \
#     --engines "DAG:causal_engine_stable" "DAG+CoT:causal_engine" \
#     --output_prefix "comparison_study" \

# python evaluation_multi.py \
#     --engines "DAG:causal_engine_stable" "DAG+CoT:causal_engine" \
#     --output_prefix "comparison_study_trainset" \
#     --test_data "/data/pclancy3/yi/26ICLR/outputs/combined_doping_data.json" # this is training set

## 0905
# Compare multiple DAG engines against Qwen
python evaluation_multi.py \
    --engines "DAG-Basic:causal_engine_stable" "DAG+CoT:causal_engine" \
    --training_graph "../outputs/filtered_combined_doping_data.json" \
    --test_data "../outputs/filtered_test_doping_data.json" \
    --output_prefix "qwen_vs_dag_comparison"

date
