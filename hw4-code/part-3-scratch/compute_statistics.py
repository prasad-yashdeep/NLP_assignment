"""
Script to compute data statistics for Q4 before and after preprocessing.
Uses T5 tokenizer to compute statistics.

WORKFLOW:
1. First run: Fill Table 1 (before preprocessing)
2. Implement T5Dataset.process_data() in load_data.py
3. Update preprocess_data() function below to match your implementation
4. Second run: Fill Table 2 (after preprocessing)
"""

import os
from transformers import T5TokenizerFast

def load_lines(path):
    """Load lines from a file."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def preprocess_data(nl_lines, sql_lines):
    """
    This matches T5Dataset.process_data() implementation
    """
    processed_nl = [f"translate to SQL: {nl}" for nl in nl_lines]
    processed_sql = sql_lines
    return processed_nl, processed_sql

def compute_statistics(nl_lines, sql_lines, tokenizer, prefix=""):
    """
    Compute statistics for natural language and SQL queries.
    
    Args:
        nl_lines: List of natural language queries
        sql_lines: List of SQL queries
        tokenizer: T5 tokenizer
        prefix: Prefix string for logging
    """
    stats = {}
    
    # Number of examples
    stats['num_examples'] = len(nl_lines)
    assert len(nl_lines) == len(sql_lines), f"NL ({len(nl_lines)}) and SQL ({len(sql_lines)}) length mismatch"
    
    # Tokenize and compute lengths
    nl_lengths = []
    sql_lengths = []
    nl_tokens_set = set()
    sql_tokens_set = set()
    
    print(f"{prefix}Processing {len(nl_lines)} examples...")
    for i, (nl, sql) in enumerate(zip(nl_lines, sql_lines)):
        if (i + 1) % 500 == 0:
            print(f"{prefix}  Processed {i + 1}/{len(nl_lines)} examples...")
        
        # Tokenize with special tokens (as will be used during training)
        nl_tokens = tokenizer.encode(nl, add_special_tokens=True)
        nl_lengths.append(len(nl_tokens))
        nl_tokens_set.update(nl_tokens)
        
        sql_tokens = tokenizer.encode(sql, add_special_tokens=True)
        sql_lengths.append(len(sql_tokens))
        sql_tokens_set.update(sql_tokens)
    
    # Mean lengths
    stats['mean_nl_length'] = sum(nl_lengths) / len(nl_lengths) if nl_lengths else 0
    stats['mean_sql_length'] = sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0
    
    # Vocabulary sizes (unique token IDs)
    stats['nl_vocab_size'] = len(nl_tokens_set)
    stats['sql_vocab_size'] = len(sql_tokens_set)
    
    # Max lengths (useful for setting max_length during training)
    stats['max_nl_length'] = max(nl_lengths) if nl_lengths else 0
    stats['max_sql_length'] = max(sql_lengths) if sql_lengths else 0
    
    return stats

def print_table(title, train_stats, dev_stats, include_num_examples=True):
    """Helper function to print statistics tables."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"\n{'Statistics Name':<45} {'Train':<20} {'Dev':<20}")
    print("-" * 85)
    
    if include_num_examples:
        print(f"{'Number of examples':<45} {train_stats['num_examples']:<20} {dev_stats['num_examples']:<20}")
    else:
        print(f"{'Model name':<45} {'google-t5/t5-small':<20} {'google-t5/t5-small':<20}")
    
    print(f"{'Mean sentence length':<45} {train_stats['mean_nl_length']:<20.2f} {dev_stats['mean_nl_length']:<20.2f}")
    print(f"{'Mean SQL query length':<45} {train_stats['mean_sql_length']:<20.2f} {dev_stats['mean_sql_length']:<20.2f}")
    print(f"{'Vocabulary size (natural language)':<45} {train_stats['nl_vocab_size']:<20} {dev_stats['nl_vocab_size']:<20}")
    print(f"{'Vocabulary size (SQL)':<45} {train_stats['sql_vocab_size']:<20} {dev_stats['sql_vocab_size']:<20}")

def main():
    # Initialize T5 tokenizer
    print("Loading T5 tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    data_folder = 'data'
    
    print("=" * 80)
    print("Q4: Data Statistics")
    print("=" * 80)
    
    # ========================================================================
    # Load raw data
    # ========================================================================
    print("\nLoading data files...")
    train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
    train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    print(f"Train: {len(train_nl)} examples")
    print(f"Dev: {len(dev_nl)} examples")
    
    # ========================================================================
    # Table 1: Statistics BEFORE Preprocessing
    # ========================================================================
    print("\n" + "=" * 80)
    print("Computing Table 1: Statistics BEFORE Preprocessing")
    print("=" * 80)
    
    print("\nComputing train statistics (BEFORE preprocessing)...")
    train_stats_before = compute_statistics(train_nl, train_sql, tokenizer, prefix="[BEFORE TRAIN] ")
    
    print("\nComputing dev statistics (BEFORE preprocessing)...")
    dev_stats_before = compute_statistics(dev_nl, dev_sql, tokenizer, prefix="[BEFORE DEV] ")
    
    # Print Table 1
    print_table("Table 1: Data Statistics BEFORE Preprocessing", 
                train_stats_before, dev_stats_before, 
                include_num_examples=True)
    
    # ========================================================================
    # Table 2: Statistics AFTER Preprocessing
    # ========================================================================
    print("\n" + "=" * 80)
    print("Computing Table 2: Statistics AFTER Preprocessing")
    print("=" * 80)
    
    # Apply preprocessing
    print("\nApplying preprocessing...")
    train_nl_processed, train_sql_processed = preprocess_data(train_nl, train_sql)
    dev_nl_processed, dev_sql_processed = preprocess_data(dev_nl, dev_sql)
    
    print("\nComputing train statistics (AFTER preprocessing)...")
    train_stats_after = compute_statistics(train_nl_processed, train_sql_processed, tokenizer, prefix="[AFTER TRAIN] ")
    
    print("\nComputing dev statistics (AFTER preprocessing)...")
    dev_stats_after = compute_statistics(dev_nl_processed, dev_sql_processed, tokenizer, prefix="[AFTER DEV] ")
    
    # Print Table 2
    print_table("Table 2: Data Statistics AFTER Preprocessing", 
                train_stats_after, dev_stats_after, 
                include_num_examples=False)
    
    # ========================================================================
    # Summary of Changes
    # ========================================================================
    print("\n" + "=" * 80)
    print("Summary of Preprocessing Impact")
    print("=" * 80)
    print(f"\n{'Metric':<50} {'Before':<15} {'After':<15} {'Change':<15}")
    print("-" * 95)
    
    # Train changes
    nl_change = train_stats_after['mean_nl_length'] - train_stats_before['mean_nl_length']
    sql_change = train_stats_after['mean_sql_length'] - train_stats_before['mean_sql_length']
    nl_vocab_change = train_stats_after['nl_vocab_size'] - train_stats_before['nl_vocab_size']
    sql_vocab_change = train_stats_after['sql_vocab_size'] - train_stats_before['sql_vocab_size']
    
    print(f"{'Train mean NL length':<50} {train_stats_before['mean_nl_length']:<15.2f} {train_stats_after['mean_nl_length']:<15.2f} {nl_change:+.2f}")
    print(f"{'Train mean SQL length':<50} {train_stats_before['mean_sql_length']:<15.2f} {train_stats_after['mean_sql_length']:<15.2f} {sql_change:+.2f}")
    print(f"{'Train NL vocab size':<50} {train_stats_before['nl_vocab_size']:<15} {train_stats_after['nl_vocab_size']:<15} {nl_vocab_change:+d}")
    print(f"{'Train SQL vocab size':<50} {train_stats_before['sql_vocab_size']:<15} {train_stats_after['sql_vocab_size']:<15} {sql_vocab_change:+d}")
    
    # Dev changes
    nl_change_dev = dev_stats_after['mean_nl_length'] - dev_stats_before['mean_nl_length']
    sql_change_dev = dev_stats_after['mean_sql_length'] - dev_stats_before['mean_sql_length']
    
    print(f"{'Dev mean NL length':<50} {dev_stats_before['mean_nl_length']:<15.2f} {dev_stats_after['mean_nl_length']:<15.2f} {nl_change_dev:+.2f}")
    print(f"{'Dev mean SQL length':<50} {dev_stats_before['mean_sql_length']:<15.2f} {dev_stats_after['mean_sql_length']:<15.2f} {sql_change_dev:+.2f}")
    
    # Additional useful statistics for hyperparameter setting
    print("\n" + "=" * 80)
    print("Additional Statistics (Useful for Training)")
    print("=" * 80)
    print(f"\nMax NL length (train): {train_stats_after['max_nl_length']} tokens")
    print(f"Max SQL length (train): {train_stats_after['max_sql_length']} tokens")
    print(f"Max NL length (dev): {dev_stats_after['max_nl_length']} tokens")
    print(f"Max SQL length (dev): {dev_stats_after['max_sql_length']} tokens")
    
    print(f"\nRecommendation: Set max_length >= {max(train_stats_after['max_sql_length'], dev_stats_after['max_sql_length'])} for SQL generation")
    
    print("\n" + "=" * 80)
    print("Statistics computation complete!")
    print("=" * 80)
    
    # Helpful reminder
    print("\nNext Steps:")
    print("1. Copy Table 1 numbers to your LaTeX report")
    if nl_change == 0 and sql_change == 0:
        print("2. WARNING: Table 2 shows no preprocessing applied yet")
        print("   - Implement T5Dataset.process_data() in load_data.py")
        print("   - Update preprocess_data() function in this script to match")
        print("   - Re-run this script to get updated Table 2")
    else:
        print("2. Copy Table 2 numbers to your LaTeX report")
        print("3. Verify preprocessing matches T5Dataset.process_data()")

if __name__ == '__main__':
    main()