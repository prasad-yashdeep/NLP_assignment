
import os
import argparse
import torch
from tqdm import tqdm
from transformers import T5TokenizerFast

from t5_utils import load_model_from_checkpoint, initialize_model
from load_data import load_t5_data, get_dataloader
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_inference_args():
    parser = argparse.ArgumentParser(description='T5 Inference')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help="Name of experiment to load")
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help="Directory containing checkpoints (optional override)")
    parser.add_argument('--num_beams', type=int, default=10,
                        help="Number of beams for generation")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size")
    parser.add_argument('--max_gen_length', type=int, default=512,
                        help="Max generation length")
    parser.add_argument('--finetune', action='store_true', default=True,
                        help="Whether model was finetuned (default True)")
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'],
                        help="Split to evaluate on")
    
    args = parser.parse_args()
    
    if args.checkpoint_dir is None:
        # Assume standard structure
        args.checkpoint_dir = os.path.join('checkpoints', 'ft_experiments', args.experiment_name)
        
    return args

def run_inference():
    args = get_inference_args()
    print(f"Loading experiment: {args.experiment_name}")
    print(f"Beams: {args.num_beams}")
    print(f"Split: {args.split}")
    
    # Load Model
    # We use 'best=True' to load best_model.pt
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Load Data
    print(f"Loading {args.split} data...")
    dataloader = get_dataloader(args.batch_size, args.split)
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    generated_queries = []
    
    print("Running generation...")
    with torch.no_grad():
        # Collate fn returns different things for dev vs test
        # Dev: encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs
        # Test: encoder_ids, encoder_mask, initial_decoder_inputs
        
        for batch in tqdm(dataloader):
            if args.split == 'dev':
                encoder_input, encoder_mask, _, _, _ = batch
            else:
                encoder_input, encoder_mask, _ = batch
                
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True
            )
            
            for gen_id in generated_ids:
                query = tokenizer.decode(gen_id, skip_special_tokens=True)
                generated_queries.append(query)
                
    # Save results
    output_sql_path = f'results/t5_beam{args.num_beams}_{args.experiment_name}_{args.split}.sql'
    output_record_path = f'records/t5_beam{args.num_beams}_{args.experiment_name}_{args.split}.pkl'
    
    save_queries_and_records(generated_queries, output_sql_path, output_record_path)
    print(f"Saved to {output_sql_path}")
    
    # If Dev, compute metrics
    if args.split == 'dev':
        gt_sql_path = 'data/dev.sql'
        gt_record_path = 'records/ground_truth_dev.pkl'
        
        sql_em, record_em, record_f1, error_msgs = compute_metrics(
            gt_sql_path, output_sql_path, gt_record_path, output_record_path
        )
        
        print("\nResults:")
        print(f"Record F1: {record_f1:.4f}")
        print(f"Record EM: {record_em:.4f}")
        print(f"SQL EM: {sql_em:.4f}")

if __name__ == "__main__":
    run_inference()


