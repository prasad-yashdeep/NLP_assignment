import argparse
import torch
import pickle
import os
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast
from t5_utils_scratch import load_model_from_checkpoint, CustomSQLTokenizer, initialize_optimizer_and_scheduler, save_model, set_seed
from dpo_utils import DPOTrainer, DPOCollator
from tqdm import tqdm
from utils import compute_metrics, save_queries_and_records
from load_data import load_t5_data
import copy

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Path to SFT checkpoint dir")
    parser.add_argument('--dpo_data', type=str, default='dpo_data.pkl')
    parser.add_argument('--experiment_name', type=str, default='dpo_experiment')
    
    parser.add_argument('--learning_rate', type=float, default=5e-6, help="Very low LR for DPO")
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10) # Usually DPO converges fast
    
    # Evaluation args (needed for eval function compatibility)
    parser.add_argument('--max_gen_length', type=int, default=128)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=16)
    
    # Dummy args for optimizer init
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--max_n_epochs', type=int, default=10)
    parser.add_argument('--scheduler_type', type=str, default="cosine")
    
    return parser.parse_args()

def eval_epoch(args, model, dev_loader, tokenizer):
    """Evaluation with custom tokenizer (Same as train_t5.py)"""
    model.eval()
    generated_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, _, _, _ in tqdm(dev_loader, desc="📊 Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True,
                pad_token_id=PAD_IDX
            )
            
            for gen_id in generated_ids:
                query = tokenizer.decode(gen_id, skip_special_tokens=True)
                generated_queries.append(query)

    # Save and evaluate
    model_sql_path = f'results/t5_dpo_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_dpo_{args.experiment_name}_dev.pkl'
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        'data/dev.sql', model_sql_path, 
        'records/ground_truth_dev.pkl', model_record_path
    )
    
    error_rate = sum(1 for msg in error_msgs if msg) / len(error_msgs)
    return record_f1, record_em, sql_em, error_rate

def main():
    set_seed(42)
    args = get_args()
    print("🚀 Starting DPO Training")
    
    # Load Tokenizer
    try:
        tokenizer = CustomSQLTokenizer()
        print("Using CustomSQLTokenizer")
    except:
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    # Load SFT Model (Policy)
    class ModelArgs:
        checkpoint_dir = args.checkpoint_dir
    
    print("Loading Policy Model...")
    policy_model = load_model_from_checkpoint(ModelArgs(), best=True)
    
    # Load Ref Model (Copy of Policy)
    print("Loading Reference Model...")
    # Ideally load from disk again to be safe, or deepcopy
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    # Load DPO Data
    with open(args.dpo_data, 'rb') as f:
        dpo_data = pickle.load(f)
    print(f"Loaded {len(dpo_data)} preference pairs")
    
    # Dataset
    collator = DPOCollator(tokenizer)
    train_loader = DataLoader(dpo_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    
    # Optimizer
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, policy_model, len(train_loader))
    
    trainer = DPOTrainer(policy_model, ref_model, optimizer, beta=args.beta)
    
    # Dev Loader for Eval
    _, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size, tokenizer)
    
    best_f1 = -1
    checkpoint_dir = f'checkpoints/dpo_{args.experiment_name}'
    
    # Training Loop
    for epoch in range(args.num_epochs):
        print(f"\n🔄 EPOCH {epoch+1}/{args.num_epochs}")
        
        total_loss = 0
        pbar = tqdm(train_loader, desc="DPO Training")
        
        for batch_idx, batch in enumerate(pbar):
            loss, stats = trainer.train_step(batch)
            total_loss += loss
            
            if scheduler:
                scheduler.step()
                
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'win_logp': f'{stats[0]:.2f}',
                    'lose_logp': f'{stats[1]:.2f}'
                })
        
        # Evaluation
        record_f1, record_em, sql_em, error_rate = eval_epoch(args, policy_model, dev_loader, tokenizer)
        print(f"🎯 F1: {record_f1:.1%} | EM: {record_em:.1%} | SQL_EM: {sql_em:.1%}")
        
        if record_f1 > best_f1:
            best_f1 = record_f1
            save_model(checkpoint_dir, policy_model, best=True)
            print(f"🏆 NEW BEST: {best_f1:.1%}")
    
    # Final Test
    print("\n🧪 Generating Final DPO Test Predictions...")
    final_model = load_model_from_checkpoint(argparse.Namespace(checkpoint_dir=checkpoint_dir), best=True)
    final_model.eval()
    
    generated_queries = []
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Test"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated_ids = final_model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True,
                pad_token_id=PAD_IDX
            )
            
            for gen_id in generated_ids:
                query = tokenizer.decode(gen_id, skip_special_tokens=True)
                generated_queries.append(query)
    
    save_queries_and_records(generated_queries, 
                            't5_dpo_experiment_test.sql',
                            't5_dpo_experiment_test.pkl')
    print("✅ DPO Training Complete")

if __name__ == "__main__":
    main()

