
import pickle
import sqlite3
import os

def check_sql_records_match(sql_file, record_file, db_path):
    with open(sql_file, 'r') as f:
        sqls = [line.strip() for line in f]
    
    with open(record_file, 'rb') as f:
        records = pickle.load(f)
    
    if len(sqls) != len(records):
        print(f"Mismatch: {len(sqls)} SQLs vs {len(records)} records")
        return

    print(f"Found {len(sqls)} entries. Checking a few...")
    
    # Connect to DB to execute SQL and verify records match
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for i in range(min(5, len(sqls))):
        sql = sqls[i]
        rec = records[i]
        print(f"\nEntry {i}:")
        print(f"SQL: {sql}")
        print(f"Record count: {len(rec)}")
        
        try:
            cursor.execute(sql)
            db_results = cursor.fetchall()
            db_results = [set(r) for r in db_results] # Normalize
            
            # This is a rough check, real eval is more complex
            print(f"Execution success. Rows returned: {len(db_results)}")
        except Exception as e:
            print(f"Execution failed: {e}")

    conn.close()

if __name__ == "__main__":
    check_sql_records_match(
        'results/t5_beam10_t5_schema_1024_linear_dev.sql',
        'records/t5_beam10_t5_schema_1024_linear_dev.pkl',
        'data/flight_database.db'
    )

