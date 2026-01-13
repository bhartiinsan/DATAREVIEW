# Multilingual Customer Review Sentiment Analysis

A comprehensive sentiment analysis research project with **multilingual NLP capabilities** for analyzing customer reviews across English and Indian languages (Hindi), using both lexicon-based (VADER) and transformer-based models (mBERT, XLM-RoBERTa, IndicBERT).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.9+-orange.svg)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.0+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Multilingual NLP Focus](#multilingual-nlp-focus)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Experimental Setup](#experimental-setup)
- [Results and Discussion](#results-and-discussion)
- [Technologies Used](#technologies-used)
- [Future Work in Multilingual NLP](#future-work-in-multilingual-nlp)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an end-to-end **multilingual sentiment analysis** pipeline that processes customer reviews across languages to:
- Classify sentiment as Positive, Negative, or Neutral
- Compare lexicon-based vs transformer-based approaches
- Handle **Indian languages** (Hindi with Devanagari script)
- Analyze cross-lingual transfer learning effectiveness
- Generate interactive visualizations and dashboards
- Perform rigorous statistical analysis with proper metrics
- Provide automated, data-driven business recommendations

**Key Highlights:** 
- **Multilingual NLP**: Supports English and Hindi, extensible to other Indian languages
- **Research-Oriented**: Proper baselines, metrics (Accuracy, F1), error analysis
- **Production-Ready**: CLI script mode, comprehensive testing, optimized performance (30-40% faster)

### 🔬 Research Focus

This project demonstrates research capabilities in applied NLP with focus on:

- **Lexicon vs Transformer Models**: Comparative analysis of rule-based (VADER) and neural approaches (mBERT, XLM-RoBERTa, IndicBERT) for sentiment classification
- **Cross-Lingual NLP**: Investigating zero-shot transfer learning from English to Indian languages (Hindi) without fine-tuning
- **Multilingual Challenges**: Addressing script differences (Devanagari), morphological complexity, and code-mixing (Hinglish)
- **Practical Customer Analytics**: Bridging research and real-world application for e-commerce feedback analysis
- **Model Evaluation**: Rigorous experimental setup with baselines, proper metrics, error analysis, and reproducible methodology

## 🌍 Multilingual NLP Focus

This project demonstrates practical multilingual NLP research capabilities, specifically targeting **Indian languages**:

### Language Support

- **English**: Full support with VADER and transformer models
- **Hindi (हिन्दी)**: Devanagari script, transformer-based sentiment analysis
- **Extensible**: Architecture supports Tamil, Telugu, Bengali, Marathi, and other Indian languages

### Cross-Lingual Challenges Addressed

1. **Script Differences**
   - Proper Unicode handling for Devanagari (Hindi)
   - Byte-pair encoding for cross-script compatibility
   - Character normalization and preprocessing pipelines

2. **Morphological Richness**
   - Subword tokenization for agglutinative languages
   - Handling compound words and inflections
   - Language-specific morphological analysis

3. **Code-Mixing (Hinglish)**
   - Future support for Hindi-English code-mixed text
   - Language identification for mixed-script reviews
   - Code-switching detection and handling

4. **Domain Adaptation**
   - E-commerce domain vocabulary
   - Indian-specific product terminology
   - Cultural context awareness

### Multilingual Models Used

- **XLM-RoBERTa**: Cross-lingual RoBERTa trained on 100 languages
- **mBERT**: Multilingual BERT covering 104 languages including Hindi
- **IndicBERT**: Specialized for 11 Indian languages (Hindi, Bengali, Tamil, etc.)

See [multilingual_sentiment.ipynb](multilingual_sentiment.ipynb) for complete multilingual experiments and cross-lingual comparison.

## ✨ Features

### Core Capabilities

- **Multilingual Sentiment Classification** for English and Hindi
- **Model Comparison**: VADER baseline vs transformer models (mBERT, XLM-R, IndicBERT)
- **Proper Research Metrics**: Accuracy, F1-score, Precision, Recall with statistical significance testing
- **Error Analysis**: Detailed failure mode analysis and model limitations
- **Interactive Dashboards** with Plotly for real-time exploration
- **Word Cloud Visualization** to identify key themes by language and sentiment
- **Statistical Analysis** including correlation and frequency analysis
- **Business Intelligence** with automated recommendation generation
- **Export Functionality** for enriched datasets and summary reports
- **CLI Script Mode**: Run analysis from command line with configurable parameters

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/bhartiinsan/DATAREVIEW.git
cd DATAREVIEW
```

2. **Install required packages**

For basic monolingual (English) analysis:
```bash
pip install -r requirements.txt
```

For multilingual analysis with transformer models:
```bash
pip install -r requirements.txt
pip install transformers torch
```

3. **Download NLTK data**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

## 💻 Usage

### Notebook Mode (Recommended for Exploration)

1. **For English sentiment analysis:**
```bash
jupyter notebook notebooks/maincode.ipynb
```

2. **For multilingual analysis (English + Hindi):**
```bash
jupyter notebook multilingual/multilingual_sentiment.ipynb
```

### Script Mode (Production/Automation)

Run the production-ready CLI script for automated analysis:

```bash
# Basic usage - English reviews with VADER
python multilingual/multilingual_analysis.py --input data/reviews_dataset.csv --output results/

# Multilingual analysis with transformers (mBERT, XLM-RoBERTa)
python multilingual/multilingual_analysis.py --input multilingual/multilingual_reviews.csv --output results/ --transformers

# Hindi-only analysis
python multilingual/multilingual_analysis.py --input multilingual/multilingual_reviews.csv --output results/ --language Hindi --transformers

# Skip visualization generation (faster)
python multilingual/multilingual_analysis.py --input data/reviews_dataset.csv --output results/ --no-plots
```

**Script Parameters:**
- `--input, -i`: Path to input CSV file with 'review' column (required)
- `--output, -o`: Output directory path (default: `output/`)
- `--language, -l`: Filter by language - `English`, `Hindi`, or `All` (default: `All`)
- `--transformers, -t`: Enable transformer models (requires `transformers` and `torch`)
- `--no-plots`: Skip generating visualization plots (faster execution)

**Output Files:**
- `sentiment_results.csv`: Full analysis results with VADER and transformer predictions
- `sentiment_distribution.png`: Visualization of sentiment distributions

### Input Data Format

Your CSV file should contain a text column with one of these names:
- `review` or `review_text` or `text` (required - the script auto-detects which one)
- `language`: Language label (optional, e.g., 'English', 'Hindi')
- `review_id`: Unique identifier (optional)
- `rating`: Numerical rating (optional)

Example:
```csv
review_id,review_text,language,rating
1,"Great product! Highly recommend.",English,5
2,"यह उत्पाद बहुत अच्छा है। सिफारिश की जाती है।",Hindi,5
3,"Terrible experience. Very disappointed.",English,1
```

## 📁 Project Structure

```
DATAREVIEW/
│
├── data/                                       # Datasets
│   └── reviews_dataset.csv                    # Sample English customer reviews
│
├── notebooks/                                  # Jupyter Notebooks
│   └── maincode.ipynb                         # Main English sentiment analysis notebook
│
├── scripts/                                    # Python Scripts
│   └── testing.py                             # Testing utilities
│
├── multilingual/                               # Multilingual NLP Research
│   ├── multilingual_sentiment.ipynb           # Hindi + English analysis notebook
│   ├── multilingual_analysis.py               # Production CLI script
│   ├── multilingual_reviews.csv               # Multilingual dataset (90 reviews)
│   ├── multilingual_sentiment_results.csv     # Analysis results
│   ├── multilingual_model_comparison.csv      # Model performance metrics
│   └── README.md                              # Multilingual module documentation
│
├── research/                                   # Research Documentation
│   ├── RESEARCH_REPORT.md                     # Complete research paper
│   ├── MSR_APPLICATION_GUIDE.md               # MSR India NLP application guide
│   ├── UPDATES_SUMMARY.md                     # Repository changelog
│   ├── sentiment_analysis_theory.md           # Theoretical foundations
│   ├── model_confidence_analysis.png          # Confidence analysis visualization
│   ├── multilingual_performance_comparison.png # Performance dashboard
│   └── README.md                              # Research folder documentation
│
├── results/                                    # Analysis Outputs
│   ├── reviews_with_sentiment_analysis.csv    # Enriched dataset with sentiment scores
│   └── sentiment_analysis_summary.csv         # Statistical summary
│
├── tests/                                      # Unit Tests
│   ├── test_data_validation.py                # Data quality validation tests
│   ├── test_sentiment_analysis.py             # Sentiment analysis tests
│   └── run_all_tests.py                       # Test suite runner
│
├── README.md                                   # Main documentation (this file)
├── requirements.txt                            # Python dependencies
├── LICENSE                                     # MIT License
└── .gitignore                                  # Git ignore rules
```

## 🔬 Methodology

### Analysis Pipeline

1. **Data Ingestion & Validation**
   - Load customer reviews from CSV
   - Validate data quality and completeness
   - Handle missing values and outliers

2. **Text Preprocessing**
   - Language detection (English vs Hindi vs code-mixed)
   - Script normalization (Unicode handling for Devanagari)
   - Text cleaning: lowercase, special character removal
   - Tokenization with language-specific rules
   - Extract review length and metadata features

3. **Sentiment Analysis**
   - **Lexicon-Based Baseline (VADER)**: English-only compound sentiment scores (-1 to +1)
   - **Transformer Models**: 
     - mBERT (Multilingual BERT)
     - XLM-RoBERTa (Cross-lingual RoBERTa)
     - IndicBERT (Indian languages specialist)
   - Classification: Positive (>0.05), Neutral (-0.05 to 0.05), Negative (<-0.05)

4. **Evaluation & Metrics**
   - Accuracy, F1-score, Precision, Recall
   - Confusion matrices by language
   - Statistical significance testing
   - Cross-lingual performance comparison

5. **Visualization & Analysis**
   - Word clouds by language and sentiment
   - Interactive Plotly dashboards
   - Performance comparison charts
   - Error analysis and failure modes

6. **Business Intelligence**
   - Automated recommendation generation
   - Export enriched datasets and reports

### VADER Sentiment Scoring

VADER (Valence Aware Dictionary and sEntiment Reasoner) baseline:
- Compound scores: -1 (most negative) to +1 (most positive)
- Context-aware (handles negations, intensifiers, emoticons)
- **Limitation**: English-only lexicon, fails on non-English text

### Transformer Models

- **mBERT**: Multilingual BERT pretrained on Wikipedia in 104 languages
- **XLM-RoBERTa**: Cross-lingual RoBERTa trained on 2.5TB of CommonCrawl data (100 languages)
- **IndicBERT**: Fine-tuned specifically for 11 Indian languages
- **Advantages**: Handle morphology, script differences, zero-shot cross-lingual transfer

## 🧪 Experimental Setup

### Dataset

- **English Reviews**: 50 customer reviews (balanced positive/negative)
- **Hindi Reviews**: 40 customer reviews with Devanagari script
- **Total**: 90 reviews across 2 languages
- **Labels**: Binary sentiment (Positive, Negative) with ratings

### Baselines

| Model | Type | Languages | Parameters |
|-------|------|-----------|------------|
| VADER | Lexicon-based | English only | Rule-based |
| mBERT | Transformer | 104 languages | 110M |
| XLM-RoBERTa | Transformer | 100 languages | 270M |
| IndicBERT | Transformer | 11 Indian languages | 110M |

### Evaluation Metrics

- **Accuracy**: Proportion of correct predictions
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Confusion Matrix**: Detailed error breakdown

## 📊 Results and Discussion

### Key Findings

1. **Multilingual Transformers Outperform Lexicon Methods**
   - XLM-RoBERTa achieves >85% accuracy on both English and Hindi
   - VADER: 90%+ on English, <50% on Hindi (as expected - English-only)
   - Transformers show consistent cross-lingual performance

2. **Cross-Lingual Transfer Works**
   - Pre-trained multilingual models handle Hindi without fine-tuning
   - Zero-shot transfer viable for Indian language sentiment analysis
   - IndicBERT shows marginal improvement over XLM-R on Hindi

3. **Language-Specific Challenges**
   - **Script**: Devanagari requires proper Unicode handling
   - **Morphology**: Hindi's agglutinative nature needs subword tokenization
   - **Code-mixing**: Real-world Hinglish remains a challenge

### Performance Table

| Language | VADER Accuracy | XLM-R Accuracy | Improvement |
|----------|---------------|----------------|-------------|
| English  | 0.920         | 0.940          | +2.2%       |
| Hindi    | 0.450         | 0.875          | +94.4%      |

*See [multilingual_sentiment.ipynb](multilingual_sentiment.ipynb) for detailed results*

### Error Analysis

**Common Failure Modes:**
- **Sarcasm**: Both models struggle with sarcastic reviews
- **Mixed sentiment**: "Good product but terrible delivery" challenging
- **Code-mixing**: Hinglish text confuses language-specific processing
- **Domain-specific terms**: Product names and technical jargon

**Limitations:**
- Small dataset (not statistically robust for publication)
- No fine-tuning (using pre-trained models only)
- Binary sentiment only (real reviews have nuance)
- No aspect-based analysis

### Output Files

1. **Sentiment Distribution**
   - Breakdown of reviews by sentiment category
   - Percentage distribution and counts

2. **Statistical Insights**
   - Correlation between review length and sentiment
   - Most frequent words in each sentiment category
   - Review length statistics by sentiment

3. **Visualizations**
   - Interactive pie charts (donut style)
   - Comprehensive 4-panel dashboard
   - Word clouds for positive/negative/neutral reviews
   - Frequency analysis bar charts

4. **Business Recommendations**
   - Automated, prioritized action items
   - Data-driven insights based on sentiment patterns

## 🛠️ Technologies Used

### Core Libraries

- **Pandas** (2.3.3+): Data manipulation and analysis
- **NumPy** (2.3.4+): Numerical computing
- **NLTK** (3.9.2+): Natural Language Processing and VADER sentiment analysis
- **Transformers** (4.0+): Hugging Face transformers library for multilingual models
- **PyTorch** (2.0+): Deep learning framework for transformer models

### Visualization

- **Matplotlib** (3.10.7+): Static plotting
- **Seaborn** (0.13.2+): Statistical visualizations
- **Plotly** (6.5.1+): Interactive dashboards
- **WordCloud** (1.9.5+): Text visualization

### Statistical Analysis

- **SciPy** (1.17.0+): Statistical testing and correlation analysis
- **scikit-learn** (1.0+): Machine learning metrics and evaluation

## 🔮 Future Work in Multilingual NLP

### Immediate Next Steps (0-3 months)

- [ ] **Expand Indian Language Coverage:**
  - [ ] Add Tamil, Telugu, Bengali, Marathi datasets
  - [ ] Compare IndicBERT vs mBERT vs XLM-R performance across all languages
  - [ ] Build language-specific preprocessing pipelines

- [ ] **Code-Mixing Support (Hinglish):**
  - [ ] Create code-mixed Hindi-English dataset
  - [ ] Implement language identification for mixed-script text
  - [ ] Test specialized code-mixed BERT models

- [ ] **Fine-Tuning Experiments:**
  - [ ] Fine-tune IndicBERT on e-commerce domain
  - [ ] Compare few-shot vs full fine-tuning strategies
  - [ ] Domain adaptation evaluation

### Medium-Term Goals (3-6 months)

- [ ] **Aspect-Based Sentiment Analysis:**
  - [ ] Extract product aspects (quality, delivery, price, service)
  - [ ] Perform fine-grained sentiment per aspect
  - [ ] Cross-lingual aspect extraction

- [ ] **Advanced Architectures:**
  - [ ] Test language-specific models (Hindi-BERT, Tamil-BERT)
  - [ ] Experiment with adapter layers for efficient fine-tuning
  - [ ] Explore distillation for production deployment

- [ ] **Robustness Testing:**
  - [ ] Adversarial examples for sarcasm, negation
  - [ ] Out-of-domain evaluation (different product categories)
  - [ ] Cross-script code-mixing challenges

### Long-Term Research (6-12 months)

- [ ] **Multimodal Sentiment:**
  - [ ] Integrate review images with text
  - [ ] Vision-language models for product reviews

- [ ] **Time Series & Trend Analysis:**
  - [ ] Track sentiment evolution over time
  - [ ] Detect emerging issues and trending topics

- [ ] **Production Deployment:**
  - [ ] Model compression (quantization, pruning, distillation)
  - [ ] Real-time sentiment API with FastAPI
  - [ ] Interactive web dashboard with Streamlit

- [ ] **Research Publication:**
  - [ ] Large-scale Indian language sentiment benchmark
  - [ ] Code-mixing challenges and solutions
  - [ ] Submit to ACL/EMNLP/NAACL workshops

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Bharti** - NLP Research & Development

## 🙏 Acknowledgments

- NLTK team for the VADER sentiment analysis tool
- Plotly team for interactive visualization capabilities
- Open source community for excellent data science libraries

---

**Note:** This project is for educational and research purposes. For production deployments, consider additional validation and testing.

**Last Updated:** 13 January 2026
