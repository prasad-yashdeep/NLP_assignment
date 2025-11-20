import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    
    text = example["text"]
    
    # QWERTY keyboard layout for typo simulation
    qwerty_neighbors = {
        'a': ['s', 'q', 'w'], 'b': ['v', 'n', 'g'], 'c': ['x', 'v', 'd'],
        'd': ['s', 'e', 'f', 'c'], 'e': ['w', 'r', 'd'], 'f': ['d', 'r', 'g'],
        'g': ['f', 't', 'h', 'b'], 'h': ['g', 'y', 'j', 'n'], 'i': ['u', 'o', 'k'],
        'j': ['h', 'u', 'k', 'm'], 'k': ['j', 'i', 'l', 'o'], 'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'], 'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'], 'q': ['w', 'a'], 'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'd', 'x'], 't': ['r', 'y', 'g'], 'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'b', 'f'], 'w': ['q', 'e', 's', 'a'], 'x': ['z', 'c', 's'],
        'y': ['t', 'u', 'h', 'g'], 'z': ['x', 'a']
    }
    
    # Tokenize the text
    tokens = word_tokenize(text)
    transformed_tokens = []
    
    for token in tokens:
        # Skip punctuation and very short tokens
        if not token.isalpha() or len(token) < 3:
            transformed_tokens.append(token)
            continue
        
        word_lower = token.lower()
        original_case = token[0].isupper() if len(token) > 0 else False
        
        # Try synonym replacement with 30% probability
        if random.random() < 0.3:
            synsets = wordnet.synsets(word_lower)
            if synsets:
                # Get all synonyms from all synsets
                synonyms = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ').lower()
                        # Filter out the original word and very different forms
                        if synonym != word_lower and len(synonym.split()) == 1:
                            synonyms.append(synonym)
                
                if synonyms:
                    # Choose a random synonym
                    chosen_synonym = random.choice(synonyms)
                    # Preserve original case
                    if original_case:
                        chosen_synonym = chosen_synonym.capitalize()
                    transformed_tokens.append(chosen_synonym)
                    continue
        
        # If no synonym replacement, try introducing typo with 20% probability
        if random.random() < 0.2 and len(word_lower) > 2:
            # Choose a random position (not first or last to keep word recognizable)
            pos = random.randint(1, len(word_lower) - 2)
            char = word_lower[pos]
            
            # Replace with a nearby QWERTY key if available
            if char in qwerty_neighbors and qwerty_neighbors[char]:
                typo_char = random.choice(qwerty_neighbors[char])
                word_with_typo = word_lower[:pos] + typo_char + word_lower[pos+1:]
                # Preserve original case
                if original_case:
                    word_with_typo = word_with_typo.capitalize()
                transformed_tokens.append(word_with_typo)
                continue
        
        # If no transformation applied, keep original word
        transformed_tokens.append(token)
    
    # Reconstruct text using TreebankWordDetokenizer
    detokenizer = TreebankWordDetokenizer()
    transformed_text = detokenizer.detokenize(transformed_tokens)
    
    example["text"] = transformed_text

    ##### YOUR CODE ENDS HERE ######

    return example
