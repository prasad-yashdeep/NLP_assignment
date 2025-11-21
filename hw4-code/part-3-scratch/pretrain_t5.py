import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from t5_utils_scratch import (initialize_model_from_scratch, initialize_optimizer_and_scheduler, 
                             save_model, CustomSQLTokenizer, set_seed)
from pretrain_data import get_pretrain_dataloader

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sql_data_path', type=str, default='data/external_train.jsonl')
    parser.add_argument('--wiki_data_path', type=str, default='data/external_wiki.jsonl')
    parser.add_argument('--experiment_name', type=str, default='pretrain_t5')
    parser.add_argument('--max_n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    
    parser.add_argument('--scheduler_type', type=str, default="cosine")
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    return parser.parse_args()

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    accumulation_steps = args.gradient_accumulation_steps
    
    pbar = tqdm(train_loader, desc="🌍 Pretraining (Multi-Task)")
    
    for batch_idx, (encoder_input, encoder_mask, decoder_input, decoder_targets) in enumerate(pbar):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

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
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler: 
                scheduler.step()

        with torch.no_grad():
            num_tokens = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * accumulation_steps * num_tokens
            total_tokens += num_tokens
            
        if batch_idx % 100 == 0:
            pbar.set_postfix({'loss': f'{total_loss/total_tokens:.4f}' if total_tokens > 0 else 'N/A'})

    return total_loss / total_tokens if total_tokens > 0 else 0

def main():
    set_seed(42)
    args = get_args()
    print("🚀 Starting Multi-Task Pretraining (SQL + General Text)")
    
    tokenizer = CustomSQLTokenizer()
    print(f"Vocab size: {tokenizer.new_vocab_size}")
    
    # Initialize model from scratch
    model = initialize_model_from_scratch(args, tokenizer)
    
    # Data
    train_loader = get_pretrain_dataloader(
        args.sql_data_path, 
        args.wiki_data_path, 
        args.batch_size, 
        tokenizer
    )
    
    # Optimizer
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    
    checkpoint_dir = f'checkpoints/{args.experiment_name}'
    
    for epoch in range(args.max_n_epochs):
        print(f"\n🔄 EPOCH {epoch+1}/{args.max_n_epochs}")
        loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"📉 Loss: {loss:.4f}")
        
        save_model(checkpoint_dir, model, best=False) # Save as last_model.pt
        if epoch % args.save_every_n_epochs == 0:
             torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch+1}.pt")

    print("✅ Pretraining Complete")

if __name__ == "__main__":
    main()
