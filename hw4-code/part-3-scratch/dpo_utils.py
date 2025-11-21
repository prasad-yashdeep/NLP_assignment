import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import copy

class DPOCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = 0  # Assuming T5 pad token ID is 0

    def __call__(self, batch):
        # batch is list of (prompt, winner, loser) tuples
        prompts = [item['prompt'] for item in batch]
        winners = [item['winner'] for item in batch]
        losers = [item['loser'] for item in batch]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            prompts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # Tokenize responses
        winner_tokens = self.tokenizer(
            winners, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        loser_tokens = self.tokenizer(
            losers, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            'input_ids': model_inputs.input_ids,
            'attention_mask': model_inputs.attention_mask,
            'winner_input_ids': winner_tokens.input_ids,
            'winner_attention_mask': winner_tokens.attention_mask,
            'loser_input_ids': loser_tokens.input_ids,
            'loser_attention_mask': loser_tokens.attention_mask
        }

class DPOTrainer:
    def __init__(self, model, ref_model, optimizer, beta=0.1):
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.beta = beta
        self.device = next(model.parameters()).device

    def get_batch_logps(self, logits, labels, attention_mask):
        """
        Compute log probabilities of labels given logits.
        """
        # Logits: [batch, seq_len, vocab]
        # Labels: [batch, seq_len]
        
        # Shift so that tokens < n predict n
        # T5 specific: labels are usually already shifted or handled by model
        # But here we are manually computing probability of sequence
        
        # For T5, the model outputs logits for decoder_input_ids. 
        # We want log p(y | x).
        
        # If using causal LM loss logic on Seq2Seq:
        # labels should be the targets.
        # logits are [batch, seq_len, vocab_size]
        
        # Mask out padding
        # labels [batch, seq]
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs of actual labels
        # shape: [batch, seq_len]
        label_log_probs = torch.gather(
            log_probs, 
            dim=2, 
            index=labels.unsqueeze(2)
        ).squeeze(2)
        
        # Mask out padding tokens
        # T5 pad token is 0
        mask = (labels != 0).float()
        
        # Sum log probs over sequence
        sum_log_probs = (label_log_probs * mask).sum(dim=1)
        
        return sum_log_probs

    def compute_loss(self, batch):
        """
        Compute DPO loss:
        L = -E [log sigmoid(beta * (log_r_w - log_r_l))]
        where r = pi_theta / pi_ref
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        winner_ids = batch['winner_input_ids'].to(self.device)
        loser_ids = batch['loser_input_ids'].to(self.device)
        
        # Forward pass for policy model (theta)
        # We need to run forward twice: once for winner, once for loser
        
        # 1. Winner Logps (Policy)
        winner_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=winner_ids
        )
        policy_winner_logps = self.get_batch_logps(winner_outputs.logits, winner_ids, batch['winner_attention_mask'].to(self.device))
        
        # 2. Loser Logps (Policy)
        loser_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=loser_ids
        )
        policy_loser_logps = self.get_batch_logps(loser_outputs.logits, loser_ids, batch['loser_attention_mask'].to(self.device))
        
        # Forward pass for reference model (ref) - NO GRAD
        with torch.no_grad():
            # 3. Winner Logps (Ref)
            ref_winner_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=winner_ids
            )
            ref_winner_logps = self.get_batch_logps(ref_winner_outputs.logits, winner_ids, batch['winner_attention_mask'].to(self.device))
            
            # 4. Loser Logps (Ref)
            ref_loser_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=loser_ids
            )
            ref_loser_logps = self.get_batch_logps(ref_loser_outputs.logits, loser_ids, batch['loser_attention_mask'].to(self.device))
        
        # DPO Loss
        # pi_theta / pi_ref
        policy_log_ratios = policy_winner_logps - policy_loser_logps
        ref_log_ratios = ref_winner_logps - ref_loser_logps
        
        logits = policy_log_ratios - ref_log_ratios
        
        losses = -F.logsigmoid(self.beta * logits)
        
        return losses.mean(), (policy_winner_logps.mean().item(), policy_loser_logps.mean().item())

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, stats = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), stats

