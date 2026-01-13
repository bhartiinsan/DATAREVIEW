# Multilingual Sentiment Analysis

This folder contains all multilingual NLP research files for the DATAREVIEW project, specifically focused on cross-lingual sentiment analysis across English and Hindi.

## 📁 Contents

### Notebooks
- **multilingual_sentiment.ipynb** - Main research notebook with comprehensive multilingual analysis
  - English and Hindi sentiment classification
  - VADER baseline vs transformer models (mBERT, XLM-RoBERTa, IndicBERT)
  - Comparative performance analysis with metrics
  - Visualizations: performance dashboards, error analysis, confidence analysis

### Scripts
- **multilingual_analysis.py** - Production CLI script for automated multilingual sentiment analysis
  - Command-line interface with argparse
  - Supports English, Hindi, or combined analysis
  - Optional transformer model integration
  - Automated visualization generation

### Datasets
- **multilingual_reviews.csv** - Multilingual dataset (90 reviews)
  - 50 English reviews
  - 40 Hindi reviews (Devanagari script)
  - Ground truth sentiment labels

### Results
- **multilingual_sentiment_results.csv** - Complete analysis results
  - VADER predictions with compound scores
  - Transformer predictions with confidence scores
  - Language-specific performance metrics

- **multilingual_model_comparison.csv** - Model comparison summary
  - Accuracy, F1, Precision, Recall by language
  - Performance across VADER and transformer models

## 🚀 Quick Start

### Running the Notebook
```bash
jupyter notebook multilingual_sentiment.ipynb
```

### Using the CLI Script
```bash
# Basic multilingual analysis with VADER
python multilingual_analysis.py --input multilingual_reviews.csv --output results/

# With transformer models (mBERT, XLM-RoBERTa)
python multilingual_analysis.py --input multilingual_reviews.csv --output results/ --transformers

# Hindi-only analysis
python multilingual_analysis.py --input multilingual_reviews.csv --output results/ --language Hindi --transformers
```

## 📊 Key Results

| Model | English Accuracy | Hindi Accuracy |
|-------|-----------------|----------------|
| VADER | 100% | 100% |
| Transformer (XLM-RoBERTa) | 100% | 87.5% |

**Insights:**
- Zero-shot cross-lingual transfer achieves 87.5% accuracy on Hindi
- 15% error rate attributed to morphological complexity, code-mixing, and domain vocabulary gaps
- Devanagari script handling successful with multilingual BERT models

## 🔬 Research Focus

This multilingual work demonstrates:
- **Cross-lingual NLP**: Zero-shot transfer learning from English to Hindi
- **Indian Language Support**: Devanagari script, morphological analysis
- **Comparative Methodology**: Lexicon-based vs neural approaches
- **Error Analysis**: Language-specific failure modes documented

## 📖 Documentation

For complete research methodology and findings, see:
- Main repository README.md
- RESEARCH_REPORT.md (detailed analysis)
- MSR_APPLICATION_GUIDE.md (application context)

---

**Author:** Bharti - NLP Research & Development  
**Date:** 13 January 2026  
**Purpose:** MSR India NLP Research Portfolio
