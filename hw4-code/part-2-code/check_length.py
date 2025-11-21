
import os
import numpy as np
from load_data import T5Dataset
from transformers import T5TokenizerFast
from prompting_utils import read_schema

def check_lengths():
    print("Checking token lengths...")
    data_folder = 'data'
    split = 'dev'
    
    # Initialize tokenizer explicitly to check specific strings
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Load Schema String directly to check its length
    schema_path = os.path.join(data_folder, 'flight_database.schema')
    schema_str = read_schema(schema_path)
    schema_tokens = tokenizer.encode(schema_str)
    print(f"Schema String Length (tokens): {len(schema_tokens)}")
    print(f"Schema String Preview: {schema_str[:200]}...")
    
    # Load Dataset
    ds = T5Dataset(data_folder, split)
    
    lengths = []
    num_truncated = 0
    
    for item in ds.data:
        enc_len = len(item['encoder_input'])
        lengths.append(enc_len)
        if enc_len >= 512:
            num_truncated += 1
            
    print(f"Total examples: {len(lengths)}")
    print(f"Average length: {np.mean(lengths):.2f}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Num truncated (>=512): {num_truncated} ({num_truncated/len(lengths)*100:.1f}%)")
    
    # Print one example of reconstructed input
    print("\nExample Input (decoded):")
    print(tokenizer.decode(ds.data[0]['encoder_input']))

if __name__ == "__main__":
    check_lengths()

