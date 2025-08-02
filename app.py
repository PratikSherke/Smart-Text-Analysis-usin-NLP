import os
import torch
import streamlit as st
import plotly.express as px
import pandas as pd
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration,
    MarianTokenizer, MarianMTModel
)

# Define paths to model directories (update these to your local paths)
BERT_PATH = "./bert_model"
T5_PATH = "./t5_model"
MARIANMT_PATH = "./marian_model"

# Function to check if model paths and files exist
def check_model_paths():
    for path in [BERT_PATH, T5_PATH, MARIANMT_PATH]:
        if not os.path.exists(path):
            st.error(f"Directory {path} not found. Ensure the model files are in the correct location.")
            raise FileNotFoundError(f"Directory {path} not found.")
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            st.error(f"Missing config.json in {path}.")
            raise FileNotFoundError(f"Missing config.json in {path}.")
        model_safetensors_path = os.path.join(path, "model.safetensors")
        model_bin_path = os.path.join(path, "pytorch_model.bin")
        if not os.path.exists(model_safetensors_path) and not os.path.exists(model_bin_path):
            st.error(f"Missing model.safetensors or pytorch_model.bin in {path}.")
            raise FileNotFoundError(f"Missing model.safetensors or pytorch_model.bin in {path}.")

# Load models and tokenizers
def load_models():
    try:
        # Load BERT for bias detection
        st.info("Loading BERT model...")
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        bert_model = BertForSequenceClassification.from_pretrained(BERT_PATH)
        
        # Load T5 for summarization
        st.info("Loading T5 model...")
        t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
        t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
        
        # Load MarianMT for translation
        st.info("Loading MarianMT model...")
        marian_tokenizer = MarianTokenizer.from_pretrained(MARIANMT_PATH)
        marian_model = MarianMTModel.from_pretrained(MARIANMT_PATH)
        
        # Move models to GPU if available
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
        st.error(f"Error loading models: {str(e)}")
        raise RuntimeError(f"Error loading models: {str(e)}")

# Bias detection with BERT
def detect_bias(text, tokenizer, model):
    labels = ['Neutral', 'Positive Bias', 'Negative Bias']
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label = labels[probs.argmax()]
    return label, probs

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

# Streamlit dashboard
def main():
    st.set_page_config(page_title="NLP Dashboard", page_icon="📊", layout="wide")
    
    # Custom CSS for colorful styling
    st.markdown("""
        <style>
        .main {background-color: #f0f2f6;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
        .stTextArea textarea {border: 2px solid #2196F3; border-radius: 5px;}
        .stMetric {background-color: #e3f2fd; border-radius: 5px; padding: 10px;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📝 Smart Text Analysis: Bias Detection, Summarization, and Translation")
    st.markdown("Enter text to analyze its bias, generate a summary, and translate it to Hindi.")
    
    # Check model paths
    try:
        check_model_paths()
    except Exception as e:
        st.error(f"Model path error: {str(e)}")
        return
    
    # Load models
    try:
        models = load_models()
        bert_tokenizer, bert_model = models["bert"]
        t5_tokenizer, t5_model = models["t5"]
        marian_tokenizer, marian_model = models["marian"]
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return
    
    # User input
    with st.form(key="input_form"):
        input_text = st.text_area(
            "Enter text to process:",
            value="Social media platforms are amplifying extreme opinions, creating a polarized environment.",
            height=150
        )
        submit_button = st.form_submit_button(label="Process Text")
    
    if submit_button and input_text:
        with st.spinner("Processing text..."):
            # Bias detection
            bias_label, bias_probs = detect_bias(input_text, bert_tokenizer, bert_model)
            st.subheader("Bias Detection")
            st.metric("Predicted Bias", bias_label)
            
            # Display probabilities in a table and bar chart
            prob_df = pd.DataFrame({
                "Label": ['Neutral', 'Positive Bias', 'Negative Bias'],
                "Probability": [f"{prob:.4f}" for prob in bias_probs]
            })
            st.write("Bias Probabilities:")
            st.table(prob_df)
            
            # Bar chart for bias probabilities
            fig = px.bar(
                prob_df,
                x="Label",
                y=bias_probs,
                title="Bias Probability Distribution",
                color="Label",
                color_discrete_map={
                    "Neutral": "#2196F3",
                    "Positive Bias": "#4CAF50",
                    "Negative Bias": "#F44336"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summarization
            summary = summarize_text(input_text, t5_tokenizer, t5_model)
            st.subheader("Summary")
            st.write(summary)
            
            # Translation
            translation = translate_text(summary, marian_tokenizer, marian_model)
            st.subheader("Translation (English to Hindi)")
            st.write(translation)

if __name__ == "__main__":
    main()