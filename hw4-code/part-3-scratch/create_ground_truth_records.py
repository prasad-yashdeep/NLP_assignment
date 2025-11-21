"""
Script to create ground truth records for dev set.
Run this ONCE before training.
"""
import os
import pickle
from utils import compute_records, read_queries

def main():
    os.makedirs('records', exist_ok=True)
    
    print("Loading dev SQL queries...")
    dev_queries = read_queries('data/dev.sql')
    print(f"Loaded {len(dev_queries)} queries")
    
    print("Computing ground truth records...")
    records, error_msgs = compute_records(dev_queries)
    
    num_errors = sum(1 for msg in error_msgs if msg != "")
    print(f"Errors: {num_errors}/{len(records)}")
    
    with open('records/ground_truth_dev.pkl', 'wb') as f:
        pickle.dump((records, error_msgs), f)
    
    print("Ground truth records saved!")

if __name__ == '__main__':
    main()