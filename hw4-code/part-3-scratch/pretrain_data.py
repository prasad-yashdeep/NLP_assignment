import json
import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

PAD_IDX = 0

class ExternalSQLDataset(Dataset):
    def __init__(self, sql_data_path, wiki_data_path, tokenizer, max_enc_len=512, max_dec_len=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        
        # Load SQL Data
        if sql_data_path and os.path.exists(sql_data_path):
            print(f"Loading SQL data from {sql_data_path}")
            with open(sql_data_path, 'r') as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line))
                    except:
                        continue
        else:
            print(f"⚠️ SQL Data path not found: {sql_data_path}")
                        
        # Load General Text Data
        if wiki_data_path and os.path.exists(wiki_data_path):
            print(f"Loading General Text data from {wiki_data_path}")
            with open(wiki_data_path, 'r') as f:
                # Optional: subsample wiki data to avoid overwhelming SQL tasks if it's huge
                # For now, we load all or a reasonable limit
                for i, line in enumerate(f):
                    if i > 200000: break # Safety limit
                    try:
                        self.data.append(json.loads(line))
                    except:
                        continue
        else:
            print(f"⚠️ Wiki Data path not found: {wiki_data_path}")
                        
        print(f"Total pretraining examples: {len(self.data)}")
        
        if len(self.data) == 0:
            raise ValueError("Dataset is empty! Check data paths and file contents.")
            
        random.shuffle(self.data)
                    
    def __len__(self):
        return len(self.data)
    
    def span_corruption(self, text):
        # Simple T5-like span corruption simulation:
        # Split text, random mask, target is masked spans.
        # Implementing full span corruption correctly is complex.
        # Alternative: Denoising objective - "denoise: [corrupted text]" -> "[original text]"
        
        # Simple approach: Prefix LM or Reconstruction
        # "denoise: {text}" -> "{text}"
        # Or better: Text Infilling (masking ~15% of tokens)
        
        # Let's do simple random span masking.
        tokens = text.split()
        if len(tokens) < 5:
            return "denoise: " + text, text
            
        # Mask one random span
        start = random.randint(0, len(tokens) - 3)
        length = random.randint(1, min(5, len(tokens)-start))
        
        masked_span = " ".join(tokens[start:start+length])
        tokens[start:start+length] = ["<extra_id_0>"]
        
        input_text = "denoise: " + " ".join(tokens)
        target_text = "<extra_id_0> " + masked_span + " <extra_id_1>"
        
        return input_text, target_text
    
    def __getitem__(self, idx):
        item = self.data[idx]
        data_type = item.get('type', 'sql')
        
        if data_type == 'sql':
            nl = item['nl']
            sql = item['sql']
            context = item.get('context', '')
            input_text = f"translate to SQL: {nl} | Schema: {context}"
            target_text = sql
        else:
            # General Text - Apply Denoising
            text = item['text']
            if len(text) > 1000: text = text[:1000] # Truncate
            input_text, target_text = self.span_corruption(text)
        
        encoder_input = self.tokenizer.encode(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_enc_len
        )
        
        decoder_output = self.tokenizer.encode(
            target_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_dec_len
        )
        
        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long)
        )

def pretrain_collate_fn(batch):
    encoder_inputs = []
    decoder_outputs = []
    
    for enc, dec in batch:
        encoder_inputs.append(enc)
        decoder_outputs.append(dec)
        
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    decoder_padded = pad_sequence(decoder_outputs, batch_first=True, padding_value=PAD_IDX)
    
    batch_size = decoder_padded.size(0)
    device = decoder_padded.device
    
    decoder_inputs = torch.cat([
        torch.zeros((batch_size, 1), dtype=torch.long, device=device),
        decoder_padded[:, :-1]
    ], dim=1)
    
    decoder_targets = decoder_padded
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets

def get_pretrain_dataloader(sql_data_path, wiki_data_path, batch_size, tokenizer):
    dset = ExternalSQLDataset(sql_data_path, wiki_data_path, tokenizer)
    return DataLoader(
        dset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=pretrain_collate_fn,
        num_workers=4
    )
