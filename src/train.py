# src/train.py
import os
import spacy
from spacy.training.example import Example
import random
import time
import pickle
from tqdm import tqdm
import mlflow
from spacy.tokens import DocBin
import mlflow.spacy

def train_ner_model(train_data_path, dev_data_path, output_dir, 
                    n_iter=30, batch_size=16, dropout=0.2):
    """Train a SpaCy NER model"""
    # Start MLflow run
    mlflow.start_run(run_name="ner_training")
    
    # Log parameters
    mlflow.log_param("n_iter", n_iter)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("dropout", dropout)
    
    # Load existing model or create new one
    print("Initializing model...")
    try:
        # Try loading a medium-sized model for better features
        nlp = spacy.load("en_core_web_md")
        print("Loaded en_core_web_md model")
    except:
        # Fallback to small model
        nlp = spacy.load("en_core_web_sm")
        print("Loaded en_core_web_sm model")
    
    # Add or get NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Load the DocBin objects
    print("Loading training data...")
    train_docbin = DocBin().from_disk(train_data_path)
    # train_docbin = spacy.load(train_data_path)
    train_docs = list(train_docbin.get_docs(nlp.vocab))
    
    print("Loading validation data...")
    dev_docbin = DocBin().from_disk(dev_data_path)
    # spacy.util.load_model(dev_data_path)
    dev_docs = list(dev_docbin.get_docs(nlp.vocab))
    
    # Get the entity labels from the data
    labels = []
    for doc in train_docs:
        for ent in doc.ents:
            labels.append(ent.label_)
    labels = list(set(labels))
    
    # Add entity labels to the NER pipe
    for label in labels:
        ner.add_label(label)
    
    # Disable other pipeline components during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    # Convert docs to Example objects for training
    train_examples = []
    for doc in tqdm(train_docs, desc="Preparing training examples"):
        train_examples.append(Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}))
    
    dev_examples = []
    for doc in tqdm(dev_docs, desc="Preparing validation examples"):
        dev_examples.append(Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}))
    
    # Training loop
    print("Starting training...")
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.begin_training()
        
        print(f"Training for {n_iter} iterations with batch size {batch_size}")
        
        # Track best model based on validation score
        best_f1 = 0.0
        
        for i in range(n_iter):
            start_time = time.time()
            
            # Shuffle examples
            random.shuffle(train_examples)
            
            # Create batches and update model
            losses = {}
            batches = spacy.util.minibatch(train_examples, size=batch_size)
            for batch in tqdm(batches, desc=f"Iteration {i+1}/{n_iter}"):
                nlp.update(batch, drop=dropout, losses=losses)
            
            # Calculate time taken
            end_time = time.time()
            duration = end_time - start_time
            
            # Evaluate on validation set
            scorer = spacy.scorer.Scorer()
            for example in dev_examples:
                pred_doc = nlp(example.reference.text)
                scorer.score(example.reference, pred_doc)
            
            # Log metrics
            metrics = {
                "ner_f": scorer.scores["ents_f"],
                "ner_p": scorer.scores["ents_p"],
                "ner_r": scorer.scores["ents_r"],
                "loss": losses["ner"]
            }
            
            mlflow.log_metrics(metrics, step=i)
            
            print(f"Iteration {i+1}/{n_iter} - Loss: {losses['ner']:.3f} - "
                  f"NER F1: {metrics['ner_f']:.3f} - "
                  f"Precision: {metrics['ner_p']:.3f} - "
                  f"Recall: {metrics['ner_r']:.3f} - "
                  f"Time: {duration:.2f}s")
            
            # Save best model
            if metrics["ner_f"] > best_f1:
                best_f1 = metrics["ner_f"]
                print(f"New best model with F1: {best_f1:.3f}")
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                nlp.to_disk(os.path.join(output_dir, "best_model"))
                mlflow.log_artifact(os.path.join(output_dir, "best_model"), "model")
        
        # Save final model
        print("Saving final model...")
        nlp.to_disk(os.path.join(output_dir, "final_model"))
        mlflow.log_artifact(os.path.join(output_dir, "final_model"), "model")
    
    mlflow.end_run()
    
    return nlp

if __name__ == "__main__":
    # Setup paths
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data", "processed")
    model_dir = os.path.join(base_dir, "models")
    
    train_data_path = os.path.join(data_dir, "train.spacy")
    dev_data_path = os.path.join(data_dir, "dev.spacy")
    output_dir = os.path.join(model_dir)
    
    # Train model
    train_ner_model(train_data_path, dev_data_path, output_dir)