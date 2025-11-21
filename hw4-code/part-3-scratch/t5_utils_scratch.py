import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config, AdamW, T5TokenizerFast
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import re
import random
import numpy as np

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed_value=42):
    """Set random seeds for reproducibility"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    transformers.set_seed(seed_value)

class CustomSQLTokenizer:
    """
    Custom SQL-aware tokenizer that better handles SQL syntax patterns.
    Addresses the assignment suggestion about tokenizer being ill-suited for SQL.
    """
    
    def __init__(self, base_tokenizer_name='google-t5/t5-small'):
        self.base_tokenizer = T5TokenizerFast.from_pretrained(base_tokenizer_name)
        
        # SQL-specific vocabulary
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'ON', 'GROUP', 'BY', 'ORDER', 'HAVING', 'DISTINCT', 'AS', 'AND', 'OR',
            'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL', 'COUNT', 'SUM', 'AVG',
            'MAX', 'MIN', 'LIMIT', 'OFFSET', 'UNION', 'ALL', 'ASC', 'DESC',
            'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'TABLE', 'DROP', 'ALTER'
        }
        
        # SQL operators and special tokens
        self.sql_operators = {
            '=': 'EQ', '!=': 'NEQ', '<>': 'NEQ', '<=': 'LTE', '>=': 'GTE',
            '<': 'LT', '>': 'GT', '(': 'LPAREN', ')': 'RPAREN',
            ',': 'COMMA', ';': 'SEMICOLON', '.': 'DOT', '*': 'STAR'
        }
        
        # Add SQL-specific tokens to vocabulary
        self._add_sql_tokens()
    
    def _add_sql_tokens(self):
        """Add SQL-specific tokens to the tokenizer vocabulary"""
        new_tokens = []
        
        # Add SQL keyword tokens
        for keyword in self.sql_keywords:
            new_tokens.append(f"SQL_KW_{keyword}")
        
        # Add SQL operator tokens  
        for op, name in self.sql_operators.items():
            new_tokens.append(f"SQL_OP_{name}")
        
        # Add common SQL patterns
        sql_patterns = [
            "SQL_TABLE", "SQL_COLUMN", "SQL_VALUE", "SQL_FUNCTION",
            "SQL_ALIAS", "SQL_SUBQUERY", "SQL_CONDITION"
        ]
        new_tokens.extend(sql_patterns)
        
        # Add tokens to tokenizer
        num_added = self.base_tokenizer.add_tokens(new_tokens)
        print(f"🔧 Added {num_added} SQL-specific tokens to vocabulary")
        
        # Store original vocab size for model initialization
        self.original_vocab_size = len(self.base_tokenizer) - num_added
        self.new_vocab_size = len(self.base_tokenizer)
    
    def preprocess_text(self, text):
        """Preprocess text to use SQL-aware tokens"""
        if not self._contains_sql(text):
            return text
        
        # Replace SQL keywords with special tokens
        for keyword in self.sql_keywords:
            pattern = r'\\b' + re.escape(keyword) + r'\\b'
            replacement = f' SQL_KW_{keyword} '
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Replace SQL operators with special tokens
        for op, name in self.sql_operators.items():
            text = text.replace(op, f' SQL_OP_{name} ')
        
        # Clean up whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        return text
    
    def postprocess_text(self, text):
        """Convert SQL tokens back to original form"""
        # Convert keywords back
        for keyword in self.sql_keywords:
            text = text.replace(f'SQL_KW_{keyword}', keyword)
        
        # Convert operators back
        for op, name in self.sql_operators.items():
            text = text.replace(f'SQL_OP_{name}', op)
        
        # Clean up spacing around operators
        text = re.sub(r'\\s*([(),;.])\\s*', r'\\1', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        return text
    
    def _contains_sql(self, text):
        """Check if text contains SQL keywords"""
        text_upper = text.upper()
        return any(keyword in text_upper for keyword in self.sql_keywords)
    
    def encode(self, text, **kwargs):
        """Encode with SQL preprocessing"""
        processed_text = self.preprocess_text(text)
        return self.base_tokenizer.encode(processed_text, **kwargs)
    
    def decode(self, token_ids, **kwargs):
        """Decode with SQL postprocessing"""
        text = self.base_tokenizer.decode(token_ids, **kwargs)
        return self.postprocess_text(text)
    
    def __len__(self):
        return len(self.base_tokenizer)

def initialize_model_from_scratch(args, tokenizer):
    """
    Initialize T5 model from scratch with custom tokenizer vocabulary.
    This addresses the assignment suggestion about custom tokenizer for SQL.
    """
    # Get base configuration
    config = T5Config.from_pretrained('google-t5/t5-small')
    
    # Update vocab size for custom tokenizer
    if hasattr(tokenizer, 'new_vocab_size'):
        config.vocab_size = tokenizer.new_vocab_size
        print(f"🔧 Model vocab size updated to: {config.vocab_size}")
    
    # Initialize model with random weights
    print("🧠 Initializing T5 from scratch (random weights)")
    model = T5ForConditionalGeneration(config)
    
    # Verify it's truly random by checking embedding weights
    first_embedding = model.shared.weight[0].sum().item()
    print(f"🔍 Random weight verification: {first_embedding:.6f}")
    
    model = model.to(DEVICE)
    
    # Print parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model initialized FROM SCRATCH:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Vocabulary size: {config.vocab_size:,}")
    
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    """
    Initialize optimizer and scheduler optimized for from-scratch training.
    Uses more conservative settings for stable long-term training.
    """
    # Calculate training steps
    effective_steps_per_epoch = epoch_length // args.gradient_accumulation_steps
    total_training_steps = effective_steps_per_epoch * args.max_n_epochs
    warmup_steps = effective_steps_per_epoch * args.num_warmup_epochs
    
    print(f"📊 Training schedule:")
    print(f"   Steps per epoch: {effective_steps_per_epoch}")
    print(f"   Total steps: {total_training_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    
    # Parameter grouping for weight decay
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() 
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() 
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # Initialize optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Initialize scheduler
    if args.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_training_steps
        )
    elif args.scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_training_steps
        )
    else:
        scheduler = None
    
    print(f"✅ Optimizer: AdamW (LR: {args.learning_rate})")
    print(f"✅ Scheduler: {args.scheduler_type}")
    
    return optimizer, scheduler

def save_model(checkpoint_dir, model, best):
    """Save model checkpoint optimized for from-scratch models"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if best:
        save_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        save_path = os.path.join(checkpoint_dir, 'last_model.pt')
    
    # Save complete model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'vocab_size': model.config.vocab_size,
    }, save_path)
    
    if best:
        print(f"   💾 Saved best model to {save_path}")

def load_model_from_checkpoint(args, best):
    """Load model checkpoint for from-scratch models"""
    checkpoint_dir = args.checkpoint_dir
    
    if best:
        load_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        load_path = os.path.join(checkpoint_dir, 'last_model.pt')
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    
    print(f"📥 Loading model from {load_path}")
    checkpoint = torch.load(load_path, map_location=DEVICE)
    
    # Recreate model with saved config
    config = checkpoint['config']
    model = T5ForConditionalGeneration(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    print(f"✅ Model loaded successfully")
    return model

def get_parameter_names(model, forbidden_layer_types):
    """Helper function for optimizer parameter grouping"""
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters
    result += list(model._parameters.keys())
    return result

# Learning rate scheduling utilities
def get_cosine_with_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1, last_epoch=-1
):
    """
    Custom cosine schedule with restarts - useful for very long training.
    Can help escape local minima during extended from-scratch training.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        if progress >= 1.0:
            return 0.0
        
        return max(
            0.0, 0.5 * (1.0 + torch.cos(torch.pi * ((float(num_cycles) * progress) % 1.0)))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def setup_advanced_training_features(args, model, optimizer):
    """
    Optional: Setup advanced features for from-scratch training
    - Gradient accumulation verification
    - Memory optimization
    - Training stability checks
    """
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled for memory efficiency")
    
    # Verify gradient accumulation setup
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"📊 Effective batch size: {effective_batch_size}")
    
    if effective_batch_size < 16:
        print("⚠️  Warning: Very small effective batch size may lead to unstable training")
    elif effective_batch_size > 64:
        print("⚠️  Warning: Very large effective batch size may slow convergence")
    
    return model

def validate_from_scratch_initialization(model):
    """
    Validate that model is truly initialized from scratch.
    Helps debug if model accidentally loads pretrained weights.
    """
    # Check a few key parameters to ensure they're random
    embedding_mean = model.shared.weight.mean().item()
    embedding_std = model.shared.weight.std().item()
    
    print(f"🔍 Initialization validation:")
    print(f"   Embedding mean: {embedding_mean:.6f}")
    print(f"   Embedding std: {embedding_std:.6f}")
    
    # Random initialization should have small mean and reasonable std
    if abs(embedding_mean) > 0.1:
        print("⚠️  Warning: Embedding mean seems high - check initialization")
    
    if embedding_std < 0.01 or embedding_std > 1.0:
        print("⚠️  Warning: Embedding std seems unusual - check initialization")
    else:
        print("✅ Initialization looks good (random)")
