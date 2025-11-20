import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    '''
    Initialize Weights & Biases for experiment tracking.
    Optional but highly recommended for tracking experiments.
    '''
    wandb.init(
        project="hw4-text-to-sql",
        name=args.experiment_name,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.max_n_epochs,
            "optimizer": args.optimizer_type,
            "scheduler": args.scheduler_type,
            "finetune": args.finetune,
        }
    )

def initialize_model(args):
    '''
    Initialize the T5 model for fine-tuning or training from scratch.
    
    Two modes:
    1. Fine-tuning (args.finetune=True): Load pretrained T5-small weights
    2. From scratch (args.finetune=False): Initialize with random weights
    
    Args:
        args: Arguments containing finetune flag
    
    Returns:
        T5ForConditionalGeneration model on the appropriate device
    '''
    if args.finetune:
        print("Loading pretrained T5-small model for fine-tuning...")
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        
        # Optional: Freeze encoder layers (uncomment if you want to try this)
        # This can speed up training and prevent overfitting on small datasets
        # for param in model.encoder.parameters():
        #     param.requires_grad = False
        # print("Encoder frozen - only training decoder")
        
    else:
        print("Initializing T5-small model from scratch...")
        config = T5Config.from_pretrained('google-t5/t5-small')
        model = T5ForConditionalGeneration(config)
        print("Model initialized with random weights")
    
    # Move model to device (GPU if available)
    model = model.to(DEVICE)
    
    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    return model

def mkdir(dirpath):
    '''Create directory if it doesn't exist.'''
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    '''
    Save model checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        model: The model to save
        best: If True, save as best model; otherwise save as last model
    '''
    mkdir(checkpoint_dir)
    
    if best:
        save_path = os.path.join(checkpoint_dir, 'best_model.pt')
        print(f"Saving best model to {save_path}")
    else:
        save_path = os.path.join(checkpoint_dir, 'last_model.pt')
    
    # Save model state dict and config
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, save_path)

def load_model_from_checkpoint(args, best):
    '''
    Load model from a checkpoint.
    
    Args:
        args: Arguments containing checkpoint_dir
        best: If True, load best model; otherwise load last model
    
    Returns:
        Loaded model on the appropriate device
    '''
    if best:
        load_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        print(f"Loading best model from {load_path}")
    else:
        load_path = os.path.join(args.checkpoint_dir, 'last_model.pt')
        print(f"Loading last model from {load_path}")
    
    checkpoint = torch.load(load_path, map_location=DEVICE)
    
    # Initialize model - if fine-tuned, load from pretrained first
    if args.finetune:
        print("Loading pretrained T5-small and applying fine-tuned weights...")
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        print("Loading model from scratch with saved config...")
        model = T5ForConditionalGeneration(checkpoint['config'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    print("Model loaded successfully")
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    '''
    Initialize optimizer and learning rate scheduler.
    
    Args:
        args: Training arguments
        model: The model to optimize
        epoch_length: Number of batches per epoch (for scheduler)
    
    Returns:
        optimizer, scheduler
    '''
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    '''
    Initialize AdamW optimizer with weight decay.
    
    Weight decay is applied to all parameters except biases and layer norms.
    This is the standard practice for transformer models.
    '''
    # Get parameters that should have weight decay
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    # Create parameter groups with and without weight decay
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

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=args.learning_rate, 
            eps=1e-8, 
            betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer_type} not implemented")

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    '''
    Initialize learning rate scheduler.
    
    Three options:
    1. none: No scheduler (constant learning rate)
    2. cosine: Cosine annealing with warmup
    3. linear: Linear decay with warmup
    '''
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        print("No learning rate scheduler")
        return None
    elif args.scheduler_type == "cosine":
        print(f"Using cosine scheduler with {num_warmup_steps} warmup steps")
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif args.scheduler_type == "linear":
        print(f"Using linear scheduler with {num_warmup_steps} warmup steps")
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler_type} not implemented")

def get_parameter_names(model, forbidden_layer_types):
    '''
    Get names of all parameters in the model, excluding certain layer types.
    
    This is used to determine which parameters should have weight decay.
    '''
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter)
    result += list(model._parameters.keys())
    return result