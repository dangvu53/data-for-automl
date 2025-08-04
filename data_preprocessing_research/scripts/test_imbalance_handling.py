#!/usr/bin/env python3
"""
Test script for imbalance handling techniques.

This script demonstrates the use of different imbalance handling techniques
on a synthetic dataset with class imbalance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import imbalance handling module
from src.preprocessing.imbalance_handling import ImbalanceHandlingPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_synthetic_dataset(n_samples=1000, imbalance_ratio=0.1):
    """
    Create a synthetic dataset with class imbalance.
    
    Args:
        n_samples (int): Total number of samples.
        imbalance_ratio (float): Ratio of minority class samples to majority class samples.
        
    Returns:
        pd.DataFrame: The synthetic dataset.
    """
    # Calculate class distribution
    n_majority = int(n_samples / (1 + imbalance_ratio * 2))
    n_minority1 = int(n_majority * imbalance_ratio)
    n_minority2 = n_samples - n_majority - n_minority1
    
    # Create class-specific texts
    majority_texts = [
        f"This is a sample text from the majority class with id {i}."
        for i in range(n_majority)
    ]
    
    minority1_texts = [
        f"This is a different text from the first minority class with id {i}."
        for i in range(n_minority1)
    ]
    
    minority2_texts = [
        f"This text belongs to the second minority class with id {i}."
        for i in range(n_minority2)
    ]
    
    # Combine texts and labels
    texts = majority_texts + minority1_texts + minority2_texts
    labels = [0] * n_majority + [1] * n_minority1 + [2] * n_minority2
    
    # Create dataframe
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the data
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def visualize_class_distribution(df, title="Class Distribution"):
    """
    Visualize the class distribution of a dataset.
    
    Args:
        df (pd.DataFrame): The dataset.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=df)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig(f"class_distribution_{title.replace(' ', '_').lower()}.png")
    plt.close()

def visualize_feature_space(df, title="Feature Space"):
    """
    Visualize the feature space of a dataset using PCA or t-SNE.
    
    Args:
        df (pd.DataFrame): The dataset.
        title (str): The title of the plot.
    """
    # Extract features
    vectorizer = CountVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['label'].values
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    
    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Class')
    plt.title(f"{title} (t-SNE Visualization)")
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f"feature_space_{title.replace(' ', '_').lower()}.png")
    plt.close()

def run_comparison(df):
    """
    Run a comparison of different imbalance handling techniques.
    
    Args:
        df (pd.DataFrame): The dataset.
    """
    # Visualize original dataset
    visualize_class_distribution(df, "Original Dataset")
    visualize_feature_space(df, "Original Dataset")
    
    # List of techniques to compare
    techniques = [
        ('random-oversampling', {}),
        ('smote', {'k_neighbors': 5}),
        ('adasyn', {'k_neighbors': 5}),
        ('eda', {'eda_alpha': 0.1, 'eda_num_aug': 4})
    ]
    
    results = []
    
    # Apply each technique and collect results
    for technique, params in techniques:
        try:
            print(f"\nApplying {technique}...")
            
            # Initialize preprocessor
            preprocessor = ImbalanceHandlingPreprocessor(
                strategy=technique,
                text_column='text',
                label_column='label',
                random_state=42,
                **params
            )
            
            # Apply transformation
            balanced_df = preprocessor.fit_transform(df)
            
            # Visualize results
            visualize_class_distribution(balanced_df, f"After {technique}")
            visualize_feature_space(balanced_df, f"After {technique}")
            
            # Collect statistics
            stats = preprocessor.get_stats()
            results.append({
                'technique': technique,
                'samples_before': stats['total_samples_before'],
                'samples_after': stats['total_samples_after'],
                'class_distribution_before': stats['class_distribution_before'],
                'class_distribution_after': stats['class_distribution_after']
            })
            
        except Exception as e:
            print(f"Error applying {technique}: {e}")
    
    # Print summary
    print("\nSummary of results:")
    for result in results:
        print(f"\nTechnique: {result['technique']}")
        print(f"Samples before: {result['samples_before']}")
        print(f"Samples after: {result['samples_after']}")
        print("Class distribution before:")
        for cls, count in result['class_distribution_before'].items():
            print(f"  Class {cls}: {count}")
        print("Class distribution after:")
        for cls, count in result['class_distribution_after'].items():
            print(f"  Class {cls}: {count}")

def main():
    """
    Main function to run the test script.
    """
    print("Creating synthetic dataset...")
    df = create_synthetic_dataset(n_samples=1000, imbalance_ratio=0.1)
    
    print(f"Dataset shape: {df.shape}")
    print("Class distribution:")
    print(df['label'].value_counts())
    
    print("\nRunning comparison of imbalance handling techniques...")
    run_comparison(df)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
