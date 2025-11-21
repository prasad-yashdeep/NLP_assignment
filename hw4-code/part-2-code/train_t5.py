import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import wandb

from t5_utils import (initialize_model, initialize_optimizer_and_scheduler, 
                      save_model, load_model_from_checkpoint, setup_wandb)
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may adjust these based on your experiments.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    # Fine-tuning is the default behavior
    parser.add_argument('--train_from_scratch', action='store_true',
                        help="Train from scratch instead of fine-tuning (default: fine-tune)")
    
    parser.add_argument('--freeze_encoder', action='store_true',
                        help="Freeze encoder parameters during fine-tuning")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", 
                        choices=["AdamW"],
                        help="Optimizer to use. Default: AdamW")
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help="Learning rate. Default: 5e-4")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="Weight decay for regularization. Default: 0.01")
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help="Gradient clipping max norm. Default: 1.0")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of gradient accumulation steps. Default: 1 (no accumulation)")

    parser.add_argument('--scheduler_type', type=str, default="cosine", 
                        choices=["none", "cosine", "linear"],
                        help="Learning rate scheduler type. Default: cosine")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="Number of warmup epochs for LR scheduler. Default: 1")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="Maximum number of training epochs. Default: 10")
    parser.add_argument('--patience_epochs', type=int, default=3,
                        help="Early stopping: wait this many epochs without improvement. Default: 3")

    parser.add_argument('--use_wandb', action='store_true',
                        help="Use Weights & Biases for experiment tracking")
    parser.add_argument('--experiment_name', type=str, default='my_first_run',
                        help="Name for this experiment. Default: my_first_run")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Training batch size. Default: 16")
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help="Evaluation batch size. Default: 16")
    
    # Generation parameters
    parser.add_argument('--max_gen_length', type=int, default=512,
                        help="Maximum length for generated SQL queries. Default: 512")
    parser.add_argument('--num_beams', type=int, default=10,
                        help="Number of beams for beam search (1 = greedy). Default: 10")

    args = parser.parse_args()
    
    # Set finetune flag (default is True, unless --train_from_scratch is passed)
    args.finetune = not args.train_from_scratch
    
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    '''
    Main training loop with evaluation and early stopping.
    '''
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    # Paths for saving dev set predictions
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'
    
    # Create directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    for epoch in range(args.max_n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.max_n_epochs}")
        print(f"{'='*60}")
        
        # Training
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Train loss: {tr_loss:.4f}")

        # Evaluation - only run after epoch 10 (i.e. starting from epoch index 9)
        if (epoch-1 )%5== 0:
            eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
                args, model, dev_loader, gt_sql_path, model_sql_path,
                gt_record_path, model_record_path
            )
            
            print(f"Dev loss: {eval_loss:.4f}")
            print(f"Dev Record F1: {record_f1:.4f}")
            print(f"Dev Record EM: {record_em:.4f}")
            print(f"Dev SQL EM: {sql_em:.4f}")
            print(f"SQL Error Rate: {error_rate*100:.2f}%")

            # Log to wandb if enabled
            if args.use_wandb:
                result_dict = {
                    'epoch': epoch,
                    'train/loss': tr_loss,
                    'dev/loss': eval_loss,
                    'dev/record_f1': record_f1,
                    'dev/record_em': record_em,
                    'dev/sql_em': sql_em,
                    'dev/error_rate': error_rate,
                }
                wandb.log(result_dict, step=epoch)

            # Early stopping logic
            if record_f1 > best_f1:
                best_f1 = record_f1
                epochs_since_improvement = 0
                print(f"New best F1: {best_f1:.4f} - Saving model")
                save_model(checkpoint_dir, model, best=True)
            else:
                epochs_since_improvement += 1
                print(f"No improvement for {epochs_since_improvement} epoch(s)")
        else:
            print("Skipping evaluation (waiting for epoch 10)")
            if args.use_wandb:
                 wandb.log({'epoch': epoch, 'train/loss': tr_loss}, step=epoch)

        # Always save last model
        save_model(checkpoint_dir, model, best=False)

        # Early stopping
        if epochs_since_improvement >= args.patience_epochs:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            print(f"Best F1: {best_f1:.4f}")
            break
    
    print(f"\nTraining complete! Best dev F1: {best_f1:.4f}")

def train_epoch(args, model, train_loader, optimizer, scheduler):
    '''
    Train for one epoch.
    
    Returns:
        Average loss per token
    '''
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        # Move to device
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
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Compute loss (only on non-padded tokens)
        # Reshape for cross entropy: [batch*seq_len, vocab_size] and [batch*seq_len]
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            decoder_targets.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # Track loss
        with torch.no_grad():
            num_tokens = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0

def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path, 
               gt_record_path, model_record_path):
    '''
    Evaluate model on dev set.
    
    Computes:
    1. Cross-entropy loss
    2. Generate SQL queries and compute metrics (F1, EM, error rate)
    
    Returns:
        eval_loss, record_f1, record_em, sql_em, error_rate
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    generated_queries = []
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_dec in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # Compute loss (for monitoring)
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )
            logits = outputs.logits

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                decoder_targets.reshape(-1)
            )
            
            num_tokens = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Generate SQL queries
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True,
            )
            
            # Decode generated queries
            for gen_id in generated_ids:
                query = tokenizer.decode(gen_id, skip_special_tokens=True)
                generated_queries.append(query)

    # Save generated queries and compute metrics
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    # Compute metrics
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    
    # Compute error rate
    num_errors = sum(1 for msg in error_msgs if msg != "")
    error_rate = num_errors / len(error_msgs) if error_msgs else 0

    eval_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    return eval_loss, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Run inference on test set and save predictions.
    '''
    model.eval()
    generated_queries = []
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    print("\nRunning inference on test set...")
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_dec in tqdm(test_loader, desc="Test inference"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            # Generate SQL queries
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True,
            )
            
            # Decode generated queries
            for gen_id in generated_ids:
                query = tokenizer.decode(gen_id, skip_special_tokens=True)
                generated_queries.append(query)

    # Save predictions
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    print(f"Test predictions saved to:")
    print(f"  SQL: {model_sql_path}")
    print(f"  Records: {model_record_path}")

def main():
    # Get arguments
    args = get_args()
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Mode: {'Fine-tuning' if args.finetune else 'Training from scratch'}")
    print(f"Device: {DEVICE}")
    
    if args.use_wandb:
        setup_wandb(args)

    # Load data
    print("\nLoading data...")
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize model
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Load best model for final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Final dev set evaluation
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'
    
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    
    print(f"\nFinal Dev Results:")
    print(f"  Loss: {dev_loss:.4f}")
    print(f"  Record F1: {dev_record_f1:.4f}")
    print(f"  Record EM: {dev_record_em:.4f}")
    print(f"  SQL EM: {dev_sql_em:.4f}")
    print(f"  Error Rate: {dev_error_rate*100:.2f}%")

    # Test set inference
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_test.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_test.pkl'
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    
    print("\nDone! Submit these files to Gradescope:")
    print(f"  {model_sql_path}")
    print(f"  {model_record_path}")

if __name__ == "__main__":
    main()