# src/evaluate.py
import os
import spacy
from spacy.scorer import Scorer
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def evaluate_model(model_path, test_data_path, output_dir):
    """Evaluate the NER model on test data"""
    print(f"Loading model from {model_path}...")
    nlp = spacy.load(model_path)
    
    print(f"Loading test data from {test_data_path}...")
    test_docbin = spacy.util.load_model(test_data_path)
    test_docs = list(test_docbin.get_docs(nlp.vocab))
    
    # Create evaluation directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Evaluate with SpaCy scorer
    print("Evaluating model...")
    scorer = Scorer()
    
    # Collect entity-level predictions for detailed analysis
    true_entities = []
    pred_entities = []
    
    # Process each document for evaluation
    for doc in test_docs:
        # True entities
        for ent in doc.ents:
            true_entities.append((ent.text, ent.label_))
        
        # Predicted entities
        pred_doc = nlp(doc.text)
        for ent in pred_doc.ents:
            pred_entities.append((ent.text, ent.label_))
        
        # Score with SpaCy scorer
        scorer.score(doc, pred_doc)
    
    # Get overall scores
    scores = scorer.scores
    
    # Save scores to JSON file
    with open(os.path.join(output_dir, "evaluation_scores.json"), "w") as f:
        json.dump({
            "ents_p": scores["ents_p"],
            "ents_r": scores["ents_r"],
            "ents_f": scores["ents_f"],
            "ents_per_type": scores["ents_per_type"]
        }, f, indent=4)
    
    # Print overall scores
    print("\nOverall Scores:")
    print(f"Precision: {scores['ents_p']:.4f}")
    print(f"Recall: {scores['ents_r']:.4f}")
    print(f"F1 Score: {scores['ents_f']:.4f}")
    
    # Print scores per entity type
    print("\nScores per Entity Type:")
    for entity_type, metrics in scores["ents_per_type"].items():
        print(f"{entity_type}:")
        print(f"  Precision: {metrics['p']:.4f}")
        print(f"  Recall: {metrics['r']:.4f}")
        print(f"  F1 Score: {metrics['f']:.4f}")
    
    # Calculate confusion matrix
    true_labels = [label for _, label in true_entities]
    pred_labels = [label for _, label in pred_entities]
    
    # Get unique entity types
    entity_types = sorted(set(true_labels) | set(pred_labels))
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(
        [entity_types.index(label) if label in entity_types else -1 for _, label in true_entities], 
        [entity_types.index(label) if label in entity_types else -1 for _, label in pred_entities], 
        labels=range(len(entity_types))
    )
    
    # Normalize confusion matrix
    norm_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, None]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_conf_matrix, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=entity_types, yticklabels=entity_types)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Generate error analysis
    errors = []
    for i, doc in enumerate(test_docs):
        gold_spans = {(ent.start_char, ent.end_char, ent.label_): ent.text for ent in doc.ents}
        pred_doc = nlp(doc.text)
        pred_spans = {(ent.start_char, ent.end_char, ent.label_): ent.text for ent in pred_doc.ents}
        
        # Find false positives
        for span, text in pred_spans.items():
            if span not in gold_spans:
                errors.append({
                    "type": "false_positive",
                    "text": text,
                    "predicted": span[2],
                    "true": "O",
                    "context": doc.text[:span[0]] + " [" + text + "] " + doc.text[span[1]:]
                })
        
        # Find false negatives
        for span, text in gold_spans.items():
            if span not in pred_spans:
                errors.append({
                    "type": "false_negative",
                    "text": text,
                    "predicted": "O",
                    "true": span[2],
                    "context": doc.text[:span[0]] + " [" + text + "] " + doc.text[span[1]:]
                })
        
        # Find mislabeled
        for span, text in gold_spans.items():
            for pred_span, pred_text in pred_spans.items():
                if span[0] == pred_span[0] and span[1] == pred_span[1] and span[2] != pred_span[2]:
                    errors.append({
                        "type": "mislabeled",
                        "text": text,
                        "predicted": pred_span[2],
                        "true": span[2],
                        "context": doc.text[:span[0]] + " [" + text + "] " + doc.text[span[1]:]
                    })
    
    # Save error analysis
    error_df = pd.DataFrame(errors)
    if not error_df.empty:
        error_df.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)
        
        # Sample errors
        print("\nSample Errors:")
        for error_type in ["false_positive", "false_negative", "mislabeled"]:
            type_errors = error_df[error_df["type"] == error_type]
            if not type_errors.empty:
                print(f"\n{error_type.replace('_', ' ').title()}:")
                for i, row in type_errors.head(3).iterrows():
                    print(f"  Text: {row['text']}")
                    print(f"  True: {row['true']}")
                    print(f"  Predicted: {row['predicted']}")
                    print(f"  Context: {row['context']}\n")
    
    return scores

if __name__ == "__main__":
    # Setup paths
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data", "processed")
    model_dir = os.path.join(base_dir, "models")
    eval_dir = os.path.join(base_dir, "evaluation")
    
    model_path = os.path.join(model_dir, "best_model")
    test_data_path = os.path.join(data_dir, "test.spacy")
    
    # Evaluate model
    evaluate_model(model_path, test_data_path, eval_dir)