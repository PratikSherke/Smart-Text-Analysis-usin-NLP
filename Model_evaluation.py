
import os
import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration,
    MarianTokenizer, MarianMTModel
)
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download NLTK data for BLEU score
nltk.download('punkt')

# Local paths
BERT_PATH = "./bert_model"
T5_PATH = "./t5_model"
MARIANMT_PATH = "./marian_model"
dataset_path = "./translated_summary_file.csv"

# Function to check if model paths and files exist
def check_model_paths():
    for path in [BERT_PATH, T5_PATH, MARIANMT_PATH]:
        print(f"Checking directory: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory {path} not found. Ensure the dataset is added correctly.")
        print(f"Directory {path} exists. Listing contents: {os.listdir(path)}")
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {path}. Found files: {os.listdir(path)}")
        model_safetensors_path = os.path.join(path, "model.safetensors")
        model_bin_path = os.path.join(path, "pytorch_model.bin")
        if not os.path.exists(model_safetensors_path) and not os.path.exists(model_bin_path):
            raise FileNotFoundError(f"Missing model.safetensors or pytorch_model.bin in {path}. Found files: {os.listdir(path)}")
        print(f"Found config.json and model file (safetensors or bin) in {path}.")

# Load models and tokenizers
def load_models():
    try:
        print(f"Loading BERT from {BERT_PATH}")
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        bert_model = BertForSequenceClassification.from_pretrained(BERT_PATH)
        
        print(f"Loading T5 from {T5_PATH}")
        t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
        t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
        
        print(f"Loading MarianMT from {MARIANMT_PATH}")
        marian_tokenizer = MarianTokenizer.from_pretrained(MARIANMT_PATH)
        marian_model = MarianMTModel.from_pretrained(MARIANMT_PATH)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model.to(device)
        t5_model.to(device)
        marian_model.to(device)
        
        return {
            "bert": (bert_tokenizer, bert_model),
            "t5": (t5_tokenizer, t5_model),
            "marian": (marian_tokenizer, marian_model)
        }
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

# Label mapping for bias detection
def map_label_to_int(label):
    label_map = {
        'Neutral': 0,
        'Positive Bias': 1,
        'Negative Bias': 2,
        0: 0,
        1: 1,
        2: 2
    }
    try:
        return label_map[label]
    except KeyError:
        raise ValueError(f"Invalid label: {label}. Expected 'Neutral', 'Positive Bias', 'Negative Bias', or integers 0, 1, 2.")

# Bias detection with BERT
def detect_bias(text, tokenizer, model):
    labels = ['Neutral', 'Positive Bias', 'Negative Bias']
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label = labels[probs.argmax()]
    return label, probs.argmax()

# Summarization with T5
def summarize_text(text, tokenizer, model):
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(model.device)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Translation with MarianMT (English to Hindi)
def translate_text(text, tokenizer, model):
    input_text = f"translate English to Hindi: {text}"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=150, truncation=True).to(model.device)
    with torch.no_grad():
        translated_ids = model.generate(**inputs)
    translation = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translation

# Evaluation functions
def evaluate_bias_detection(texts, true_labels, tokenizer, model):
    pred_labels = []
    true_labels_int = [map_label_to_int(label) for label in true_labels]
    
    for text in texts:
        _, pred_label_idx = detect_bias(text, tokenizer, model)
        pred_labels.append(pred_label_idx)
    
    f1 = f1_score(true_labels_int, pred_labels, average='weighted')
    report = classification_report(true_labels_int, pred_labels, target_names=['Neutral', 'Positive Bias', 'Negative Bias'])
    return f1, report

def evaluate_summarization(texts, reference_summaries, tokenizer, model):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for text, ref_summary in zip(texts, reference_summaries):
        pred_summary = summarize_text(text, tokenizer, model)
        scores = scorer.score(ref_summary, pred_summary)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            rouge_scores[metric].append(scores[metric].fmeasure)
    
    avg_scores = {metric: np.mean(scores) for metric, scores in rouge_scores.items()}
    return avg_scores

def evaluate_translation(summaries, reference_translations, tokenizer, model):
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    
    for summary, ref_translation in zip(summaries, reference_translations):
        pred_translation = translate_text(summary, tokenizer, model)
        ref_tokens = [nltk.word_tokenize(ref_translation)]
        pred_tokens = nltk.word_tokenize(pred_translation)
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)
    
    avg_bleu = np.mean(bleu_scores)
    return avg_bleu

# Main function to process and evaluate
def process_and_evaluate():
    try:
        # Check model paths
        check_model_paths()
        
        # Load models
        models = load_models()
        bert_tokenizer, bert_model = models["bert"]
        t5_tokenizer, t5_model = models["t5"]
        marian_tokenizer, marian_model = models["marian"]
        
        # Load test dataset and sample 30%
        # dataset_path = "/kaggle/input/dataset-file/translated_summary_file.csv"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Test dataset not found at {dataset_path}.")
        
        df = pd.read_csv(dataset_path)
        required_columns = ['article', 'label', 'summary', 'summary_hindi']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}. Found: {list(df.columns)}")
        
        # Sample 30% of the dataset
        df_sampled = df.sample(frac=0.3, random_state=42)
        print(f"Using {len(df_sampled)} samples (30% of {len(df)} total samples) for evaluation.")
        
        texts = df_sampled['article'].tolist()
        true_bias_labels = df_sampled['label'].tolist()
        reference_summaries = df_sampled['summary'].tolist()
        reference_translations = df_sampled['summary_hindi'].tolist()
        
        # Validate labels
        try:
            true_bias_labels = [map_label_to_int(label) for label in true_bias_labels]
        except ValueError as e:
            raise ValueError(f"Error in label column: {str(e)}")
        
        # Evaluate bias detection
        print("\nEvaluating Bias Detection...")
        f1, report = evaluate_bias_detection(texts, true_bias_labels, bert_tokenizer, bert_model)
        print(f"F1-Score (weighted): {f1:.4f}")
        print("Classification Report:")
        print(report)
        
        # Evaluate summarization
        print("\nEvaluating Summarization...")
        rouge_scores = evaluate_summarization(texts, reference_summaries, t5_tokenizer, t5_model)
        print("ROUGE Scores:")
        for metric, score in rouge_scores.items():
            print(f"  {metric}: {score:.4f}")
        
        # Evaluate translation
        print("\nEvaluating Translation...")
        bleu_score = evaluate_translation(reference_summaries, reference_translations, marian_tokenizer, marian_model)
        print(f"BLEU Score: {bleu_score:.4f}")
        
        # Process a single example for demonstration
        print("\nProcessing a single example...")
        text = input("Enter text to process (or press Enter for default): ").strip()
        if not text:
            text = "Social media platforms are amplifying extreme opinions, creating a polarized environment."
        
        bias_label, _ = detect_bias(text, bert_tokenizer, bert_model)
        summary = summarize_text(text, t5_tokenizer, t5_model)
        translation = translate_text(summary, marian_tokenizer, marian_model)
        
        print(f"\nPredicted Bias: {bias_label}")
        print(f"Summary: {summary}")
        print(f"Translation (en to hi): {translation}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Run the pipeline
if __name__ == "__main__":
    process_and_evaluate()