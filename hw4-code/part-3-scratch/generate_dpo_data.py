import argparse
import torch
import os
import pickle
from tqdm import tqdm
from transformers import T5TokenizerFast
from t5_utils_scratch import load_model_from_checkpoint, CustomSQLTokenizer
from load_data import load_t5_data
from utils import compute_metrics, compute_record
import random

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='extra_credit_long')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='dpo_data.pkl')
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--num_candidates', type=int, default=4) # Generate N candidates
    return parser.parse_args()

def main():
    args = get_args()
    print(f"Generating DPO data from {args.checkpoint_dir}")

    # Load tokenizer and model
    # We can't easily check if CustomSQLTokenizer was used just from args here, 
    # but our plan uses it.
    try:
        tokenizer = CustomSQLTokenizer()
        print("Using CustomSQLTokenizer")
    except:
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        print("Using Default Tokenizer")

    # Mock args for load_model
    class ModelArgs:
        checkpoint_dir = args.checkpoint_dir
    
    model = load_model_from_checkpoint(ModelArgs(), best=True)
    model.eval()

    # Load training data (we use training data to generate self-play preference pairs)
    # We only need the training set
    train_loader, _, _ = load_t5_data(batch_size=1, test_batch_size=1, tokenizer=tokenizer)

    preference_pairs = []
    
    # Process a subset to save time if needed, but full pass is better
    for i, batch in enumerate(tqdm(train_loader, desc="Generating Pairs")):
        encoder_ids, encoder_mask, decoder_inputs, decoder_targets, _ = batch
        
        encoder_ids = encoder_ids.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        
        # Ground truth SQL
        gt_ids = decoder_targets[0]
        gt_ids = gt_ids[gt_ids != 0] # Remove padding
        gt_sql = tokenizer.decode(gt_ids, skip_special_tokens=True)
        
        # Get Ground Truth execution result
        _, gt_record, gt_error = compute_record("gt", gt_sql)
        
        if gt_error:
            continue # Skip invalid GT

        # Generate candidates using sampling or diversity beam search
        # Here we use standard beam search and take top K
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=encoder_ids,
                attention_mask=encoder_mask,
                max_length=128,
                num_beams=args.num_beams,
                num_return_sequences=args.num_candidates,
                early_stopping=True,
                pad_token_id=0
            )
        
        candidates = []
        for gen_id in generated_ids:
            cand_sql = tokenizer.decode(gen_id, skip_special_tokens=True)
            candidates.append(cand_sql)
            
        # Evaluate candidates
        valid_candidates = []
        invalid_candidates = []
        
        for cand_sql in candidates:
            _, cand_record, cand_error = compute_record("cand", cand_sql)
            
            # Check if correct
            is_correct = False
            if not cand_error:
                # Check Record Match
                if set(cand_record) == set(gt_record):
                    is_correct = True
            
            if is_correct:
                valid_candidates.append(cand_sql)
            else:
                invalid_candidates.append(cand_sql)
        
        # Form pairs
        # Strategy: 
        # Pair GT (Winner) vs Invalid Candidate (Loser)
        # Pair Valid Candidate (Winner) vs Invalid Candidate (Loser)
        
        prompt = tokenizer.decode(encoder_ids[0], skip_special_tokens=True)
        
        # 1. GT vs Generated Loser
        for loser in invalid_candidates:
            preference_pairs.append({
                'prompt': prompt,
                'winner': gt_sql,
                'loser': loser
            })
            
        # 2. Generated Winner vs Generated Loser (Self-Correction)
        # Limit to 1 pair per prompt to avoid imbalance
        if valid_candidates and invalid_candidates:
            winner = random.choice(valid_candidates)
            loser = random.choice(invalid_candidates)
            if winner != gt_sql: # Avoid duplicate if we already added GT
                 preference_pairs.append({
                    'prompt': prompt,
                    'winner': winner,
                    'loser': loser
                })

    print(f"Generated {len(preference_pairs)} preference pairs")
    
    with open(args.output_file, 'wb') as f:
        pickle.dump(preference_pairs, f)
        
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()

