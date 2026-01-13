"""
Extract and save visualization images from maincode.ipynb to images folder.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/reviews_dataset.csv')

# Preprocessing
print("Preprocessing data...")
df['clean_review'] = df['review_text'].str.lower()
df['review_length'] = df['review_text'].str.len()

# Calculate sentiment (simple VADER-like approach)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment_category'] = df['sentiment_score'].apply(
    lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
)

COLOR_MAP = {
    'Positive': '#2ecc71',
    'Negative': '#e74c3c',
    'Neutral': '#f39c12'
}

print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# ========== Visualization 1: Review Length Distribution ==========
print("\n1. Generating review_length_distribution.png...")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['review_length'], bins=30, kde=True, ax=ax, color='steelblue')
ax.set_title("Review Length Distribution", fontsize=14, fontweight='bold')
ax.set_xlabel("Review Length (characters)")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig('images/review_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: review_length_distribution.png")

# ========== Visualization 2: Word Clouds ==========
print("\n2. Generating sentiment_wordclouds.png...")
categories = ['Positive', 'Neutral', 'Negative']
color_schemes = {'Positive': 'Greens', 'Neutral': 'Oranges', 'Negative': 'Reds'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, category in enumerate(categories):
    category_reviews = df[df['sentiment_category'] == category]['clean_review'].dropna()
    reviews_text = " ".join(category_reviews)
    
    if reviews_text.strip():
        wordcloud = WordCloud(
            width=600, 
            height=400, 
            background_color='white',
            colormap=color_schemes[category],
            max_words=40,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(reviews_text)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(
            f'{category} Reviews\n({len(category_reviews)} reviews)', 
            fontsize=14, 
            fontweight='bold'
        )
    else:
        axes[idx].text(
            0.5, 0.5, 
            f'No {category} reviews found', 
            ha='center', 
            va='center', 
            fontsize=14,
            color='gray'
        )
    
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('images/sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: sentiment_wordclouds.png")

# ========== Visualization 3: Top Words Bar Chart ==========
print("\n3. Generating top_words_by_sentiment.png...")

def get_top_words(text, n=10):
    from collections import Counter
    words = text.split()
    # Simple stopwords
    stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'this', 'that', 'it', 'from'}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(words).most_common(n)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, category in enumerate(['Positive', 'Negative', 'Neutral']):
    subset = df[df['sentiment_category'] == category]
    if len(subset) > 0:
        reviews_text = " ".join(subset['clean_review'])
        top_words = get_top_words(reviews_text, 10)
        
        words = [word for word, count in top_words]
        counts = [count for word, count in top_words]
        
        axes[idx].barh(words, counts, color=COLOR_MAP[category], alpha=0.8, edgecolor='black')
        axes[idx].set_title(f'{category} Reviews\n(Top 10 Words)', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Frequency', fontsize=11)
        axes[idx].invert_yaxis()
        axes[idx].grid(axis='x', alpha=0.3)
    else:
        axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
        axes[idx].set_title(f'{category} Reviews', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('images/top_words_by_sentiment.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: top_words_by_sentiment.png")

print("\n" + "="*60)
print("✓ All visualizations generated successfully!")
print("="*60)
print("\nGenerated images:")
print("  1. images/review_length_distribution.png")
print("  2. images/sentiment_wordclouds.png")
print("  3. images/top_words_by_sentiment.png")
