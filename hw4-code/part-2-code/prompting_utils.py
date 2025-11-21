import os


import json
import nltk
from nltk.corpus import stopwords
from collections import Counter

def read_schema(schema_path):
    '''
    Read the .schema file and return a compact string representation.
    Format: table_name (col1, col2, ...)
    '''
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    schema_str_parts = []
    
    # The schema structure has "types", "ents", "defaults", "links"
    # "ents" seems to contain the table definitions
    # Let's map the "ents" to a compact string
    
    if 'ents' in schema:
        for table_name, columns in schema['ents'].items():
            col_names = [col_name for col_name in columns.keys()]
            # Compact format: table(col1,col2) - remove spaces to save tokens
            schema_str_parts.append(f"{table_name}({','.join(col_names)})")
    
    return ",".join(schema_str_parts)

def load_alignment(alignment_path):
    '''
    Load alignment file into a dictionary.
    Format in file: "phrase \t db_value"
    Returns: dict {phrase: db_value} sorted by length of phrase (descending) to prioritize longer matches
    '''
    alignment = {}
    if os.path.exists(alignment_path):
        with open(alignment_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    phrase = parts[0].lower()
                    db_value = parts[1]
                    alignment[phrase] = db_value
    
    # Sort keys by length descending to match longest phrases first
    sorted_alignment = dict(sorted(alignment.items(), key=lambda item: len(item[0]), reverse=True))
    return sorted_alignment

class ExampleRetriever:
    def __init__(self, train_nl, train_sql):
        self.train_nl = train_nl
        self.train_sql = train_sql
        self.stopwords = set(stopwords.words('english'))
        # Precompute sets of words for Jaccard similarity
        self.train_sets = [self._tokenize(s) for s in train_nl]

    def _tokenize(self, text):
        # Simple tokenization and stopword removal
        words = text.lower().split()
        return {w for w in words if w not in self.stopwords}

    def retrieve(self, query, k=2):
        query_set = self._tokenize(query)
        scores = []
        
        for i, train_set in enumerate(self.train_sets):
            # Jaccard Similarity: intersection / union
            intersection = len(query_set.intersection(train_set))
            union = len(query_set.union(train_set))
            score = intersection / union if union > 0 else 0
            
            # Avoid retrieving the exact same query (if testing on train set)
            if score < 0.99: 
                scores.append((score, i))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        top_k_indices = [idx for _, idx in scores[:k]]
        return [(self.train_nl[idx], self.train_sql[idx]) for idx in top_k_indices]

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")