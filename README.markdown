# AI-Powered Text Insights: Bias Analysis, Summarization, and Translation for News Media


## 📝 Overview

**AI-Powered Text Insights** is a cutting-edge Natural Language Processing (NLP) application designed to analyze newspaper articles for bias, generate concise summaries, and translate content into Hindi. Leveraging fine-tuned transformer models—BERT, T5, and MarianMT—this project delivers robust media analysis through a user-friendly Streamlit dashboard with interactive Plotly visualizations. The application showcases advanced NLP techniques, model fine-tuning, and seamless integration for real-world use cases.

This project demonstrates expertise in:
- Fine-tuning transformer models for task-specific performance.
- Developing interactive web applications with Streamlit.
- Utilizing PyTorch and Hugging Face for scalable NLP solutions.
- Visualizing model outputs for actionable insights.

## ✨ Features

- **Bias Detection**: Fine-tuned BERT model classifies text as Neutral, Positive Bias, or Negative Bias, with probability visualizations.
- **Text Summarization**: Fine-tuned T5 model generates concise summaries of input articles.
- **Translation**: Translates summaries from English to Hindi using a fine-tuned MarianMT model.
- **Interactive Dashboard**: Streamlit-based UI for intuitive text input and result exploration.
- **Performance Metrics**: Models evaluated with industry-standard metrics (see Model Performance section).

## 📊 Model Performance

The models were fine-tuned on task-specific datasets, and their performance was rigorously evaluated:

- **BERT (Bias Detection)**:
  - **Accuracy**: 0.92
  - **Precision**: 0.90
  - **Recall**: 0.91
  - **F1-Score**: 0.90
- **T5 (Summarization)**:
  - **ROUGE-1**: 0.85
  - **ROUGE-2**: 0.60
  - **ROUGE-L**: 0.78
- **MarianMT (Translation)**:
  - **BLEU Score**: 0.82

Fine-tuning scripts is available as Model_evaluation python file in the repostitory. Datasets for fine tuning and training are available in the repository as 'Data file'.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional, for faster inference)
- Required Python packages (listed in `requirements.txt`)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ai-text-insights.git
   cd ai-text-insights
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Fine-Tuned Models**:
   - Place the fine-tuned model directories (`bert_model`, `t5_model`, `marian_model`) in the project root.
   - Alternatively, download pre-trained models from Hugging Face and fine-tune them using the scripts in the `finetuning` directory.

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

### Directory Structure
```
ai-text-insights/
├── app.py                                                          # Main Streamlit application
├── Data file                                                       # Datasets for fine-tuning and training 
├── text summarizer fine tune main
    text translation and text bias det fine tune main/              # Scripts for model fine-tuning
├── Model_evaluation                                                # script for model evaluation                                       
├── requirements.txt         # Python dependencies              
└── README.md                # Project documentation
```

## 🛠️ Usage

1. **Launch the Application**:
   Run `streamlit run app.py` to start the Streamlit dashboard.

2. **Input Text**:
   Enter a newspaper article or text in the provided text area.

3. **Analyze**:
   Click the "Process Text" button to:
   - Detect bias with a probability distribution (visualized as a bar chart).
   - Generate a concise summary of the input text.
   - Translate the summary into Hindi.

4. **View Results**:
   Results are displayed in an interactive dashboard with bias metrics, summaries, and translations.

## 🔧 Technologies Used

- **Python**: Core programming language.
- **Hugging Face Transformers**: BERT, T5, and MarianMT for NLP tasks.
- **PyTorch**: Model training and inference.
- **Streamlit**: Interactive web application framework.
- **Plotly**: Data visualization for bias probabilities.
- **Pandas**: Data handling for processing and visualization.

## 📈 Future Enhancements

- Support for additional translation languages.
- Integration with real-time news APIs for dynamic article analysis.
- Enhanced visualizations with additional metrics.
- Cloud deployment on platforms like AWS or Heroku.

