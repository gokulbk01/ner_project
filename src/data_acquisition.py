# src/data_acquisition.py
import os
import requests
import pandas as pd
import re
from tqdm import tqdm

def download_conll2003():
    """Download CoNLL-2003 dataset from a reliable source"""
    data_dir = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for the CoNLL-2003 dataset files
    urls = {
        "train": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train",
        "dev": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa",
        "test": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb"
    }
    
    file_paths = {}
    
    for split, url in urls.items():
        file_path = os.path.join(data_dir, f"conll2003_{split}.txt")
        file_paths[split] = file_path
        
        if not os.path.exists(file_path):
            print(f"Downloading {split} dataset...")
            response = requests.get(url)
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {split} dataset to {file_path}")
        else:
            print(f"{split} dataset already exists at {file_path}")
    
    return file_paths

def parse_conll_file(file_path):
    """Parse CoNLL-2003 format file into sentences with tagged entities"""
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Empty line indicates end of sentence
            if not line or line.startswith('-DOCSTART-'):
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            # Parse token, POS tag, chunk tag, and NER tag
            parts = line.split(' ')
            if len(parts) >= 4:
                token, pos, chunk, ner = parts[0], parts[1], parts[2], parts[3]
                current_sentence.append((token, pos, chunk, ner))
    
    # Add the last sentence if it's not empty
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def convert_to_spacy_format(sentences):
    """Convert CoNLL sentences to SpaCy's format for NER training"""
    spacy_data = []
    
    for sentence in tqdm(sentences, desc="Converting to SpaCy format"):
        text = " ".join([token[0] for token in sentence])
        entities = []
        
        # Extract entity spans
        i = 0
        while i < len(sentence):
            token, _, _, ner = sentence[i]
            
            # Check if this token starts an entity
            if ner.startswith('B-'):
                entity_type = ner[2:]  # Remove the B- prefix
                start_char = len(" ".join([t[0] for t in sentence[:i]]))
                if i > 0:
                    start_char += 1  # Add space
                
                # Find where this entity ends
                end_idx = i
                for j in range(i + 1, len(sentence)):
                    if sentence[j][3].startswith('I-') and sentence[j][3][2:] == entity_type:
                        end_idx = j
                    else:
                        break
                
                # Calculate end character position
                end_char = start_char + len(" ".join([t[0] for t in sentence[i:end_idx + 1]]))
                
                entities.append((start_char, end_char, entity_type))
                i = end_idx + 1
            else:
                i += 1
        
        spacy_data.append((text, {"entities": entities}))
    
    return spacy_data

def prepare_conll_data():
    """Download and prepare CoNLL-2003 dataset for NER training"""
    file_paths = download_conll2003()
    
    # Process each split
    processed_data = {}
    for split, file_path in file_paths.items():
        print(f"Processing {split} dataset...")
        sentences = parse_conll_file(file_path)
        processed_data[split] = convert_to_spacy_format(sentences)
        
        # Save processed data
        processed_dir = os.path.join(os.getcwd(), "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        import pickle
        with open(os.path.join(processed_dir, f"spacy_{split}.pickle"), "wb") as f:
            pickle.dump(processed_data[split], f)
        
        print(f"Processed {len(processed_data[split])} sentences for {split}")
    
    return processed_data

if __name__ == "__main__":
    prepare_conll_data()