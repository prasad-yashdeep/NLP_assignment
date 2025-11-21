#!/bin/bash

# Activate Conda Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/yp2693/NLP_assignment/hw4-code/part-2-code/penv_part2

set -e # Exit on error

# Define Absolute Paths
WORKSPACE_ROOT="/scratch/yp2693/NLP_assignment"
SCRATCH_DIR="$WORKSPACE_ROOT/hw4-code/part-3-scratch"
DATA_DIR="$SCRATCH_DIR/data"

echo "========================================================"
echo "STEP 0: Process External Data"
echo "========================================================"
python process_external_data.py

echo "========================================================"
echo "STEP 1: Pretraining T5 on External Data (SQL + General Text)"
echo "========================================================"
# Train on the large external dataset + WikiText first
python pretrain_t5.py \
    --sql_data_path "$DATA_DIR/external_train.jsonl" \
    --wiki_data_path "$DATA_DIR/external_wiki.jsonl" \
    --experiment_name t5_pretrain_multitask \
    --max_n_epochs 5 \
    --learning_rate 5e-4

echo "========================================================"
echo "STEP 2: Fine-Tuning on Assignment Data (SFT Phase)"
echo "========================================================"
# Resume from pretrained checkpoint
python train_t5.py \
    --train_from_scratch \
    --use_sql_tokenizer \
    --max_n_epochs 25 \
    --patience_epochs 5 \
    --experiment_name sft_finetuned_v2 \
    --resume_from_checkpoint checkpoints/t5_pretrain_multitask/last_model.pt \
    --resume_epoch 0 

echo "========================================================"
echo "STEP 3: Generating DPO Preference Pairs"
echo "========================================================"
python generate_dpo_data.py \
    --checkpoint_dir checkpoints/extra_credit_experiments/sft_finetuned_v2 \
    --output_file dpo_data_v2.pkl \
    --num_beams 10 \
    --num_candidates 4

echo "========================================================"
echo "STEP 4: DPO Training (RL Phase)"
echo "========================================================"
python train_dpo.py \
    --checkpoint_dir checkpoints/extra_credit_experiments/sft_finetuned_v2 \
    --dpo_data dpo_data_v2.pkl \
    --experiment_name dpo_final_v2 \
    --beta 0.1 \
    --learning_rate 1e-5 \
    --num_epochs 10

echo "========================================================"
echo "PIPELINE COMPLETE"
echo "========================================================"
