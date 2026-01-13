#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilingual Sentiment Analysis - Production CLI Script
MSR India NLP Research Project

This script performs multilingual sentiment analysis on customer reviews
using both lexicon-based (VADER) and transformer-based models (mBERT, XLM-RoBERTa).

Author: Bharti - NLP Research & Development
Date: January 2026
"""

import argparse
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Core libraries
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# NLP libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: transformers library not available. Install with: pip install transformers torch")

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MultilingualSentimentAnalyzer:
    """
    Production-ready multilingual sentiment analysis pipeline.
    
    Supports:
    - English and Hindi (Devanagari script)
    - VADER lexicon-based baseline
    - Transformer models (mBERT, XLM-RoBERTa, IndicBERT)
    - Comprehensive metrics and error analysis
    """
    
    def __init__(self, use_transformers: bool = True):
        """Initialize sentiment analyzers."""
        print("🔧 Initializing Multilingual Sentiment Analyzer...")
        
        # Download NLTK data
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            print("✅ VADER lexicon loaded")
        except Exception as e:
            print(f"❌ Error loading VADER: {e}")
            sys.exit(1)
        
        # Load transformer model
        self.sentiment_pipeline = None
        if use_transformers and TRANSFORMERS_AVAILABLE:
            try:
                print("📥 Loading transformer model (this may take a few minutes on first run)...")
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    truncation=True,
                    max_length=512
                )
                print("✅ Transformer model loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not load transformer model: {e}")
                print("   Continuing with VADER-only analysis...")
    
    def predict_vader(self, text: str) -> str:
        """Predict sentiment using VADER lexicon."""
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def predict_transformer(self, text: str) -> Optional[str]:
        """Predict sentiment using transformer model."""
        if self.sentiment_pipeline is None:
            return None
        
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            label = result['label']
            
            # Map star ratings to sentiment
            if '5 stars' in label or '4 stars' in label:
                return 'Positive'
            elif '1 star' in label or '2 stars' in label:
                return 'Negative'
            else:  # 3 stars
                return 'Neutral'
        except Exception as e:
            print(f"⚠️  Transformer prediction error: {e}")
            return None
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'review', 
                         language_column: str = 'language') -> pd.DataFrame:
        """
        Analyze sentiment for entire dataframe.
        
        Args:
            df: Input dataframe with reviews
            text_column: Name of column containing text
            language_column: Name of column containing language labels
            
        Returns:
            DataFrame with sentiment predictions and confidence scores
        """
        print(f"\n📊 Analyzing {len(df)} reviews...")
        
        results = []
        for idx, row in df.iterrows():
            text = row[text_column]
            language = row.get(language_column, 'Unknown')
            
            # VADER prediction
            vader_pred = self.predict_vader(text)
            vader_scores = self.sia.polarity_scores(text)
            
            # Transformer prediction
            trans_pred = None
            trans_conf = None
            if self.sentiment_pipeline:
                trans_pred = self.predict_transformer(text)
                if trans_pred:
                    raw_result = self.sentiment_pipeline(text[:512])[0]
                    trans_conf = raw_result['score']
            
            results.append({
                'review': text,
                'language': language,
                'vader_sentiment': vader_pred,
                'vader_compound': vader_scores['compound'],
                'transformer_sentiment': trans_pred,
                'transformer_confidence': trans_conf
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(df)} reviews...")
        
        print("✅ Analysis complete!")
        return pd.DataFrame(results)
    
    def compute_metrics(self, df: pd.DataFrame, 
                       true_column: str = 'true_sentiment',
                       pred_column: str = 'vader_sentiment') -> Dict:
        """Compute classification metrics."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        
        y_true = df[true_column]
        y_pred = df[pred_column]
        
        # Remove None values
        valid_mask = y_pred.notna()
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multilingual Sentiment Analysis for Customer Reviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze English reviews only
  python multilingual_analysis.py --input reviews.csv --output results/
  
  # Analyze multilingual dataset with transformer models
  python multilingual_analysis.py --input multilingual_reviews.csv --output results/ --transformers
  
  # Hindi-only analysis
  python multilingual_analysis.py --input reviews.csv --output results/ --language Hindi
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with reviews (must have "review" column)')
    parser.add_argument('--output', '-o', default='output/',
                       help='Output directory for results (default: output/)')
    parser.add_argument('--language', '-l', choices=['English', 'Hindi', 'All'],
                       default='All', help='Filter by language (default: All)')
    parser.add_argument('--transformers', '-t', action='store_true',
                       help='Use transformer models (requires transformers library)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  MULTILINGUAL SENTIMENT ANALYSIS - MSR India NLP Research")
    print("=" * 70)
    print(f"\n📁 Input: {args.input}")
    print(f"📂 Output: {output_dir}")
    print(f"🌐 Language: {args.language}")
    print(f"🤖 Transformers: {'Enabled' if args.transformers else 'Disabled'}")
    
    # Load data
    try:
        df = pd.read_csv(args.input)
        print(f"\n✅ Loaded {len(df)} reviews from {args.input}")
        
        # Validate required columns - accept 'review', 'review_text', or 'text'
        text_column = None
        for col in ['review', 'review_text', 'text']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            print("❌ Error: Input CSV must have one of: 'review', 'review_text', or 'text' column")
            print(f"   Found columns: {df.columns.tolist()}")
            sys.exit(1)
        
        # Rename to standard 'review' column
        if text_column != 'review':
            df = df.rename(columns={text_column: 'review'})
            print(f"   Using '{text_column}' column for review text")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Filter by language if specified
    if args.language != 'All' and 'language' in df.columns:
        df = df[df['language'] == args.language].copy()
        print(f"   Filtered to {len(df)} {args.language} reviews")
    
    # Initialize analyzer
    analyzer = MultilingualSentimentAnalyzer(use_transformers=args.transformers)
    
    # Perform analysis
    results_df = analyzer.analyze_dataframe(df)
    
    # Save results
    output_file = output_dir / 'sentiment_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("  ANALYSIS SUMMARY")
    print("=" * 70)
    
    if 'language' in results_df.columns:
        print("\n📊 Language Distribution:")
        print(results_df['language'].value_counts())
    
    print("\n📊 VADER Sentiment Distribution:")
    print(results_df['vader_sentiment'].value_counts())
    
    if args.transformers and results_df['transformer_sentiment'].notna().any():
        print("\n📊 Transformer Sentiment Distribution:")
        print(results_df['transformer_sentiment'].value_counts())
    
    # Generate plots
    if not args.no_plots and 'vader_sentiment' in results_df.columns:
        print("\n📊 Generating visualizations...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # VADER sentiment distribution
        results_df['vader_sentiment'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('VADER Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sentiment')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Transformer sentiment distribution
        if args.transformers and results_df['transformer_sentiment'].notna().any():
            results_df['transformer_sentiment'].value_counts().plot(kind='bar', ax=axes[1], color='coral')
            axes[1].set_title('Transformer Sentiment Distribution', fontsize=14, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'Transformers not enabled', ha='center', va='center', fontsize=12)
            axes[1].set_title('Transformer Sentiment Distribution', fontsize=14, fontweight='bold')
        
        axes[1].set_xlabel('Sentiment')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_file = output_dir / 'sentiment_distribution.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   📈 Plot saved to: {plot_file}")
        plt.close()
    
    print("\n" + "=" * 70)
    print("  ✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults available in: {output_dir}")
    

if __name__ == '__main__':
    main()
