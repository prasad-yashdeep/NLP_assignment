import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO: Implement this
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Process the data
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        '''
        Load and preprocess the data.
        
        Steps:
        1. Load .nl and .sql files
        2. Add task prefix to natural language input (common T5 practice)
        3. Tokenize both input and output
        4. Return list of dictionaries with tokenized data
        '''
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = self.load_lines(nl_path)
        
        # Load SQL queries (not available for test set)
        sql_path = os.path.join(data_folder, f'{split}.sql')
        if os.path.exists(sql_path):
            sql_lines = self.load_lines(sql_path)
        else:
            sql_lines = [None] * len(nl_lines)  # Test set has no labels
        
        processed_data = []
        
        for nl, sql in zip(nl_lines, sql_lines):
            # Preprocessing: Add task prefix (helps T5 understand the task)
            # This is a common practice for T5 models
            nl_processed = f"translate to SQL: {nl}"
            
            # Tokenize natural language input (for encoder)
            encoder_input = tokenizer.encode(
                nl_processed,
                add_special_tokens=True,
                truncation=True,
                max_length=512  # T5-small can handle up to 512 tokens
            )
            
            # Tokenize SQL output (for decoder) - only if available
            if sql is not None:
                decoder_output = tokenizer.encode(
                    sql,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512
                )
            else:
                decoder_output = None
            
            processed_data.append({
                'encoder_input': encoder_input,
                'decoder_output': decoder_output,
                'nl_text': nl,  # Keep original for debugging
                'sql_text': sql
            })
        
        return processed_data
    
    def load_lines(self, path):
        '''Helper function to load lines from a file.'''
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        return lines
    
    def __len__(self):
        '''Return the number of examples in the dataset.'''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Get a single example from the dataset.
        
        Returns:
            For train/dev: (encoder_input, decoder_output)
            For test: (encoder_input, None)
        '''
        item = self.data[idx]
        
        if self.split == 'test':
            # Test set: only return encoder input
            return torch.tensor(item['encoder_input'], dtype=torch.long), None
        else:
            # Train/dev: return both encoder input and decoder output
            return (
                torch.tensor(item['encoder_input'], dtype=torch.long),
                torch.tensor(item['decoder_output'], dtype=torch.long)
            )


def normal_collate_fn(batch):
    '''
    Collation function for training and evaluation (train/dev sets).
    
    This function:
    1. Pads sequences to the same length within the batch
    2. Creates attention masks
    3. Prepares decoder inputs and targets for teacher forcing
    
    Inputs:
        batch: List of tuples (encoder_input, decoder_output) from __getitem__
    
    Returns:
        encoder_ids: Padded encoder inputs [batch_size, max_enc_len]
        encoder_mask: Attention mask for encoder [batch_size, max_enc_len]
        decoder_inputs: Padded decoder inputs [batch_size, max_dec_len]
        decoder_targets: Padded decoder targets [batch_size, max_dec_len]
        initial_decoder_inputs: First token for generation [batch_size, 1]
    '''
    encoder_inputs = []
    decoder_outputs = []
    
    for encoder_input, decoder_output in batch:
        encoder_inputs.append(encoder_input)
        decoder_outputs.append(decoder_output)
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # Create encoder attention mask (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Pad decoder outputs
    decoder_padded = pad_sequence(decoder_outputs, batch_first=True, padding_value=PAD_IDX)
    
    # For teacher forcing in T5:
    # decoder_inputs: prepend pad_token_id (0), then all tokens except the last (eos)
    # decoder_targets: the full sequence (what we want to predict)
    batch_size = decoder_padded.size(0)
    device = decoder_padded.device
    
    # Prepend pad_token_id to create decoder input
    decoder_inputs = torch.cat([
        torch.zeros((batch_size, 1), dtype=torch.long, device=device),  # pad_token_id at start
        decoder_padded[:, :-1]  # all but last token (eos)
    ], dim=1)
    
    # Decoder targets are the full sequence
    decoder_targets = decoder_padded
    
    # Initial decoder input (just pad_token_id for generation during eval)
    initial_decoder_inputs = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function for test set inference.
    
    For test set, we only have encoder inputs (no ground truth SQL).
    
    Inputs:
        batch: List of tuples (encoder_input, None) from __getitem__
    
    Returns:
        encoder_ids: Padded encoder inputs [batch_size, max_enc_len]
        encoder_mask: Attention mask for encoder [batch_size, max_enc_len]
        initial_decoder_inputs: Starting token for generation [batch_size, 1]
    '''
    encoder_inputs = []
    
    for encoder_input, _ in batch:
        encoder_inputs.append(encoder_input)
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # Create encoder attention mask
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder input: T5 uses pad token (0) as the start token
    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.zeros((batch_size, 1), dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    '''
    Create a DataLoader for the specified split.
    
    Args:
        batch_size: Batch size for the dataloader
        split: One of 'train', 'dev', or 'test'
    
    Returns:
        DataLoader object
    '''
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"  # Only shuffle training data
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    '''
    Load all three dataloaders.
    
    Args:
        batch_size: Batch size for training
        test_batch_size: Batch size for dev/test
    
    Returns:
        train_loader, dev_loader, test_loader
    '''
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    '''Load lines from a file (kept for compatibility).'''
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    '''
    Load data for prompting task (optional extra credit with Gemma).
    This is used in prompting.py, not for T5 fine-tuning.
    '''
    train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
    train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_nl = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_nl, train_sql, dev_nl, dev_sql, test_nl