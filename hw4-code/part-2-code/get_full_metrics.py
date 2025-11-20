#!/usr/bin/env python3
"""
Script to get full evaluation metrics for dev set results.
"""
from utils import compute_metrics

# Paths
pred_sql = "results/t5_ft_my_first_run_dev.sql"
pred_records = "records/t5_ft_my_first_run_dev.pkl"
dev_sql = "data/dev.sql"
dev_records = "records/ground_truth_dev.pkl"

# Compute all metrics
sql_em, record_em, record_f1, error_msgs = compute_metrics(
    dev_sql, pred_sql, dev_records, pred_records
)

# Compute error rate
num_errors = sum(1 for msg in error_msgs if msg != "")
error_rate = num_errors / len(error_msgs) if error_msgs else 0

# Print results
print("=" * 60)
print("Development Set Evaluation Results")
print("=" * 60)
print(f"Record F1 Score:        {record_f1:.4f} ({record_f1*100:.2f}%)")
print(f"Record Exact Match:     {record_em:.4f} ({record_em*100:.2f}%)")
print(f"SQL Query Exact Match:  {sql_em:.4f} ({sql_em*100:.2f}%)")
print(f"SQL Error Rate:         {error_rate:.4f} ({error_rate*100:.2f}%)")
print(f"Total Queries:          {len(error_msgs)}")
print(f"Queries with Errors:    {num_errors}")
print("=" * 60)

