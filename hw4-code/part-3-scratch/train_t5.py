#!/usr/bin/env python3
"""
T5 From-Scratch Training with Custom SQL Tokenization - Extra Credit
====================================================================

Optimized for 80+ epoch training to reach ≥50% F1 for 1.5% extra credit.
Includes custom SQL-aware tokenization as suggested in the assignment.
"""

import os
import argparse
import re
from tqdm import tqdm
import torch
import torch.nn as nn

from t5_utils_scratch import (initialize_model_from_scratch, initialize_optimizer_and_scheduler, 
                             save_model, load_model_from_checkpoint, CustomSQLTokenizer, set_seed)
from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    """Arguments optimized for from-scratch training with custom tokenization"""
    parser = argparse.ArgumentParser(description='T5 From-Scratch Training - Extra Credit')
    
    # FORCE from-scratch training
    parser.add_argument('--train_from_scratch', action='store_true', default=True)
    
    # Custom tokenization
    parser.add_argument('--use_sql_tokenizer', action='store_true', default=True,
                        help="Use custom SQL-aware tokenization")
    
    # Optimized hyperparameters for long from-scratch training
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help="Lower LR for stable long training")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=0.5,
                        help="Conservative gradient clipping")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Large effective batch size")
    
    parser.add_argument('--scheduler_type', type=str, default="cosine")
    parser.add_argument('--num_warmup_epochs', type=int, default=5,
                        help="Long warmup for from-scratch")
    parser.add_argument('--max_n_epochs', type=int, default=25,
                        help="Long training for from-scratch")
    parser.add_argument('--patience_epochs', type=int, default=5,
                        help="Very patient for slow from-scratch learning")
    
    parser.add_argument('--experiment_name', type=str, default='extra_credit_long')
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Small batch for memory efficiency")
    parser.add_argument('--test_batch_size', type=int, default=16)
    
    # Generation parameters
    parser.add_argument('--max_gen_length', type=int, default=512)
    parser.add_argument('--num_beams', type=int, default=10)  # Greedy for speed
    
    # Evaluation frequency (less frequent for speed and stability)
    parser.add_argument('--eval_every_n_epochs', type=int, default=5,
                        help="Evaluate every N epochs to save time and reduce interruptions")
    parser.add_argument('--save_every_n_epochs', type=int, default=10)
    
    # Add resume functionality
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help="Epoch to resume from")
    
    args = parser.parse_args()
    args.finetune = False  # FORCE from-scratch
    return args

def train_epoch(args, model, train_loader, optimizer, scheduler):
    """Training epoch with gradient accumulation for stable learning"""
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    accumulation_steps = args.gradient_accumulation_steps
    
    pbar = tqdm(train_loader, desc="🔄 Training")
    
    for batch_idx, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(pbar):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Forward pass
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )
        
        loss = criterion(
            outputs.logits.reshape(-1, outputs.logits.size(-1)),
            decoder_targets.reshape(-1)
        ) / accumulation_steps
        
        loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler: 
                scheduler.step()

        # Track metrics
        with torch.no_grad():
            num_tokens = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * accumulation_steps * num_tokens
            total_tokens += num_tokens
            
        # Update progress bar
        if batch_idx % 100 == 0:
            pbar.set_postfix({'loss': f'{total_loss/total_tokens:.4f}' if total_tokens > 0 else 'N/A'})

    return total_loss / total_tokens if total_tokens > 0 else 0

def eval_epoch(args, model, dev_loader, tokenizer):
    """Evaluation with custom tokenizer"""
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
    model_sql_path = f'results/t5_scr_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_scr_{args.experiment_name}_dev.pkl'
    
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
    
    print("🚀 T5 EXTRA CREDIT: Training from scratch with Custom SQL Tokenization")
    print("="*80)
    print(f"🎯 Target: ≥50% F1 for 1.5% extra credit")
    print(f"⏰ Long training: {args.max_n_epochs} epochs")
    print(f"🔧 Custom SQL tokenizer: {'YES' if args.use_sql_tokenizer else 'NO'}")
    print(f"💻 Device: {DEVICE}")
    
    # Initialize custom tokenizer BEFORE loading data
    if args.use_sql_tokenizer:
        tokenizer = CustomSQLTokenizer()
        print(f"📊 Vocab expanded: {tokenizer.original_vocab_size} → {tokenizer.new_vocab_size}")
    else:
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    # Load data with tokenizer
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size, tokenizer)
    
    # Initialize model with custom vocab size
    model = initialize_model_from_scratch(args, tokenizer)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    
    print(f"📈 Training config:")
    print(f"   LR: {args.learning_rate}")
    print(f"   Batch: {args.batch_size}x{args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print(f"   Eval every: {args.eval_every_n_epochs} epochs")
    print("="*80)
    
    # Training loop with resume capability
    best_f1 = -1
    patience = 0
    start_epoch = args.resume_epoch
    checkpoint_dir = f'checkpoints/extra_credit_experiments/{args.experiment_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"📥 Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state restored")
        
        if 'best_f1' in checkpoint:
            best_f1 = checkpoint['best_f1']
            print(f"✅ Previous best F1: {best_f1:.1%}")
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✅ Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.max_n_epochs):
        print(f"\n🔄 EPOCH {epoch+1}/{args.max_n_epochs}")
        print("-" * 60)
        
        # Train
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"📉 Loss: {tr_loss:.4f}")
        
        # Evaluate less frequently to save time
        if epoch % args.eval_every_n_epochs == 0 or epoch == args.max_n_epochs - 1:
            record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader, tokenizer)
            
            print(f"🎯 F1: {record_f1:.1%} | EM: {record_em:.1%} | SQL_EM: {sql_em:.1%} | Error: {error_rate:.1%}")
            
            if record_f1 > best_f1:
                best_f1 = record_f1
                patience = 0
                save_model(checkpoint_dir, model, best=True)
                print(f"🏆 NEW BEST: {best_f1:.1%} {'🎉 TARGET REACHED!' if best_f1 >= 0.5 else ''}")
            else:
                patience += 1
                
        # Save checkpoint periodically
        if epoch % args.save_every_n_epochs == 0:
            save_model(checkpoint_dir, model, best=False)
            
        # Early stopping (adjusted patience for 25-epoch eval frequency)
        if patience >= 2:  # 2 evaluations * 25 epochs = 50 epochs without improvement
            print(f"🛑 Early stopping - no improvement for {patience * args.eval_every_n_epochs} epochs")
            break
    
    # Final test inference
    print(f"\n🧪 Generating final test predictions...")
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    generated_queries = []
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="📝 Test"):
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
    
    # Save submission files
    save_queries_and_records(generated_queries, 
                            't5_ft_experiment_ec_test.sql',
                            't5_ft_experiment_ec_test.pkl')
    
    print(f"\n✅ EXTRA CREDIT TRAINING COMPLETE!")
    print("="*60)
    print(f"🏆 Best F1: {best_f1:.1%}")
    print(f"🔧 Custom SQL tokenization: Used")
    print(f"🎯 Extra Credit (≥50%): {'ACHIEVED! 🎉' if best_f1 >= 0.5 else 'Keep training...'}")
    print(f"📤 Submission files:")
    print(f"  • t5_ft_experiment_ec_test.sql")
    print(f"  • t5_ft_experiment_ec_test.pkl")

if __name__ == "__main__":
    main()
