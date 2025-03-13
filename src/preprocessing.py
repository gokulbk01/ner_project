# src/preprocessing.py
import os
import pickle
import spacy
from spacy.tokens import DocBin
import random
from tqdm import tqdm

def load_data():
    """Load the processed CoNLL data"""
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    data = {}
    
    for split in ["train", "dev", "test"]:
        with open(os.path.join(processed_dir, f"spacy_{split}.pickle"), "rb") as f:
            data[split] = pickle.load(f)
    
    return data["train"], data["dev"], data["test"]

def convert_to_spacy_docbin(data, nlp, output_path):
    """Convert data to SpaCy DocBin format for efficient loading during training"""
    db = DocBin()
    
    for text, annotations in tqdm(data, desc=f"Converting to DocBin ({output_path})"):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    
    db.to_disk(output_path)
    print(f"Saved DocBin to {output_path}")

def preprocess_data():
    """Preprocess the data and convert it to SpaCy's DocBin format"""
    train_data, dev_data, test_data = load_data()
    
    # Initialize SpaCy
    nlp = spacy.blank("en")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DocBin format
    convert_to_spacy_docbin(train_data, nlp, os.path.join(output_dir, "train.spacy"))
    convert_to_spacy_docbin(dev_data, nlp, os.path.join(output_dir, "dev.spacy"))
    convert_to_spacy_docbin(test_data, nlp, os.path.join(output_dir, "test.spacy"))
    
    # Also save the data splits for easy access later
    splits = {
        "train": train_data,
        "dev": dev_data,
        "test": test_data
    }
    with open(os.path.join(output_dir, "data_splits.pickle"), "wb") as f:
        pickle.dump(splits, f)
    
    return splits

if __name__ == "__main__":
    preprocess_data()