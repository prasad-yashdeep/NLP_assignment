import pandas as pd
import json
import os

WORKSPACE_ROOT = "/scratch/yp2693/NLP_assignment"

def convert_parquet_to_jsonl(parquet_file, output_file, text_column='text', format_type='sql'):
    print(f"Converting {parquet_file} to {output_file}...")
    if not os.path.exists(parquet_file):
        print(f"❌ Source file not found: {parquet_file}")
        return False
        
    try:
        df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(df)} rows.")
        
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w') as f:
            for _, row in df.iterrows():
                if format_type == 'sql':
                    nl = row.get('sql_prompt', '')
                    context = row.get('sql_context', '')
                    sql = row.get('sql', '')
                    
                    if not nl or not sql:
                        continue
                    
                    record = {
                        "nl": nl,
                        "sql": sql,
                        "context": context,
                        "type": "sql"
                    }
                elif format_type == 'general':
                    text = row.get(text_column, '')
                    if not text:
                        continue
                    
                    record = {
                        "text": text,
                        "type": "general"
                    }

                f.write(json.dumps(record) + "\n")
                
        print(f"✅ Conversion complete: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error converting {parquet_file}: {e}")
        return False

if __name__ == "__main__":
    data_dir = os.path.join(WORKSPACE_ROOT, "temp_data_download")
    output_dir = os.path.join(WORKSPACE_ROOT, "hw4-code/part-3-scratch/data")
    
    # 1. SQL Data
    train_parquet = os.path.join(data_dir, "synthetic_text_to_sql_train.snappy.parquet")
    convert_parquet_to_jsonl(train_parquet, os.path.join(output_dir, "external_train.jsonl"), format_type='sql')
    
    test_parquet = os.path.join(data_dir, "synthetic_text_to_sql_test.snappy.parquet")
    convert_parquet_to_jsonl(test_parquet, os.path.join(output_dir, "external_extra.jsonl"), format_type='sql')

    # 2. General Text Data (WikiText)
    wikitext_parquet = os.path.join(data_dir, "wikitext-103-train.parquet")
    convert_parquet_to_jsonl(wikitext_parquet, os.path.join(output_dir, "external_wiki.jsonl"), format_type='general')
