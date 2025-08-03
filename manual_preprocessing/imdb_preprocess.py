#!/usr/bin/env python3
"""
Load dataset from Hugging Face and train AutoGluon TextPredictor
Dataset: MothMalone/data-preprocessing-automl-benchmarks
Uses AutoGluon's TextPredictor for optimal text classification performance
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from autogluon.tabular import TabularPredictor
import os
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"

def load_huggingface_dataset(dataset_name="MothMalone/data-preprocessing-automl-benchmarks"):
    """
    Load dataset from Hugging Face Hub
    
    Parameters
    ----------
    dataset_name : str
        The Hugging Face dataset identifier
        
    Returns
    -------
    tuple
        (train_df, test_df) pandas DataFrames
    """
    print("🔄 Loading dataset from Hugging Face Hub...")
    print(f"   Dataset: {dataset_name}")
    
    start_time = time.time()
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name, "imdb")
        
        # Check available splits
        available_splits = list(dataset.keys())
        print(f"   Available splits: {available_splits}")
        
        # Convert to pandas DataFrames
        train_df = None
        test_df = None
        
        if 'train' in available_splits:
            train_df = pd.DataFrame(dataset['train'])
            print(f"   Train set loaded: {len(train_df)} samples")
        
        if 'test' in available_splits:
            test_df = pd.DataFrame(dataset['test'])
            print(f"   Test set loaded: {len(test_df)} samples")
        elif 'validation' in available_splits:
            test_df = pd.DataFrame(dataset['validation'])
            print(f"   Validation set loaded (using as test): {len(test_df)} samples")
        
        # If no test set, create one from train
        if test_df is None and train_df is not None:
            from sklearn.model_selection import train_test_split
            print("   No test set found, splitting train set...")
            
            train_df, test_df = train_test_split(
                train_df, 
                test_size=0.2, 
                random_state=42,
                stratify=train_df.get('label', None)
            )
            print(f"   Train split: {len(train_df)} samples")
            print(f"   Test split: {len(test_df)} samples")
        
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded successfully in {format_time(load_time)}")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None, None

def explore_dataset(train_df, test_df):
    """
    Explore and analyze the dataset structure
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    """
    print("\n" + "="*60)
    print("📊 Dataset Exploration")
    print("="*60)
    
    if train_df is not None:
        print(f"\n🔍 Training Set Analysis:")
        print(f"   Shape: {train_df.shape}")
        print(f"   Columns: {list(train_df.columns)}")
        
        # Check data types
        print(f"\n   Data Types:")
        for col, dtype in train_df.dtypes.items():
            print(f"     {col}: {dtype}")
        
        # Look for text columns
        text_columns = [col for col in train_df.columns if train_df[col].dtype == 'object']
        print(f"\n   Text columns: {text_columns}")
        
        # Look for target column
        potential_targets = ['label', 'target', 'class', 'y']
        target_col = None
        for col in potential_targets:
            if col in train_df.columns:
                target_col = col
                break
        
        if target_col:
            print(f"\n   Target column: {target_col}")
            print(f"   Target distribution:")
            target_counts = train_df[target_col].value_counts()
            for value, count in target_counts.items():
                percentage = count / len(train_df) * 100
                print(f"     {value}: {count} ({percentage:.1f}%)")
        else:
            print(f"   ⚠️ No obvious target column found")
            print(f"   Available columns: {list(train_df.columns)}")
        
        # Show sample data
        print(f"\n   Sample data:")
        print(train_df.head(3).to_string())
        
        # Text statistics for text columns
        for col in text_columns:
            if col != target_col:
                print(f"\n   Text statistics for '{col}':")
                text_lengths = train_df[col].astype(str).str.len()
                print(f"     Length - Min: {text_lengths.min()}, Max: {text_lengths.max()}, Mean: {text_lengths.mean():.1f}")
                
                # Show sample texts
                print(f"     Sample texts:")
                for i, text in enumerate(train_df[col].head(2)):
                    print(f"       [{i+1}] {str(text)[:100]}{'...' if len(str(text)) > 100 else ''}")
    
    if test_df is not None:
        print(f"\n🔍 Test Set Analysis:")
        print(f"   Shape: {test_df.shape}")
        print(f"   Columns: {list(test_df.columns)}")

def train_autogluon_classifier(train_df, test_df, target_column='label', time_limit=1800):
    """
    Train AutoGluon text classifier using TextPredictor
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    target_column : str
        Name of the target column
    time_limit : int
        Training time limit in seconds
        
    Returns
    -------
    TextPredictor
        Trained AutoGluon text predictor
    """
    print("\n" + "="*60)
    print("🤖 AutoGluon Text Classifier Training")
    print("="*60)
    
    if train_df is None:
        print("❌ No training data available")
        return None
    
    # Validate target column
    if target_column not in train_df.columns:
        print(f"❌ Target column '{target_column}' not found!")
        print(f"   Available columns: {list(train_df.columns)}")
        return None
    
    # Find text column
    text_columns = [col for col in train_df.columns if train_df[col].dtype == 'object' and col != target_column]
    if not text_columns:
        print("❌ No text columns found for text classification!")
        return None
    
    text_column = text_columns[0]  # Use first text column
    print(f"   Using text column: {text_column}")
    
    # Create model directory
    model_path = f"autogluon_text_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_path, exist_ok=True)
    
    print(f"\n⚙️ Training Configuration:")
    print(f"   Text column: {text_column}")
    print(f"   Target column: {target_column}")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Time limit: {format_time(time_limit)}")
    print(f"   Model path: {model_path}")
    
    # Determine problem type
    num_classes = train_df[target_column].nunique()
    problem_type = 'binary' if num_classes == 2 else 'multiclass'
    print(f"   Problem type: {problem_type} ({num_classes} classes)")
    
    # Create TextPredictor
    predictor = TabularPredictor(
        label=target_column,
        path=model_path,
        problem_type=problem_type,
        eval_metric='accuracy',
        verbosity=2
    )
    
    # Start training
    print(f"\n🏋️ Starting training at {datetime.now().strftime('%H:%M:%S')}...")
    train_start = time.time()
    
    try:
        # Train with TextPredictor - it automatically handles text preprocessing
        
        predictor.fit(
            train_data=train_df,
            time_limit=time_limit,
            hyperparameter_tune_kwargs='auto',
            presets=['best_quality'],  # Use best quality for text classification
        )
        
        train_time = time.time() - train_start
        print(f"✅ Training completed in {format_time(train_time)}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None
    
    # Show model leaderboard
    print(f"\n📋 Model Performance:")
    try:
        # For TextPredictor, we'll evaluate on training data
        train_score = predictor.evaluate(train_df, metrics=['accuracy'])
        print(f"Training accuracy: {train_score['accuracy']:.4f}")
        
    except Exception as e:
        print(f"⚠️ Could not evaluate model: {e}")
    
    return predictor

def evaluate_model(predictor, test_df, target_column='label'):
    """
    Evaluate the trained TextPredictor model on test data
    
    Parameters
    ----------
    predictor : TextPredictor
        Trained AutoGluon text predictor
    test_df : pd.DataFrame
        Test data
    target_column : str
        Name of the target column
    """
    print("\n" + "="*60)
    print("📈 Model Evaluation")
    print("="*60)
    
    if predictor is None or test_df is None:
        print("❌ Missing predictor or test data")
        return
    
    print(f"\n🔮 Making predictions on {len(test_df)} test samples...")
    pred_start = time.time()
    
    try:
        # Prepare test data
        if target_column in test_df.columns:
            # Test data has labels - we can evaluate
            # Use TextPredictor's built-in evaluation
            test_scores = predictor.evaluate(test_df, metrics=['accuracy', 'f1'])
            
            print(f"✅ Test Results:")
            for metric, score in test_scores.items():
                print(f"   {metric}: {score:.4f}")
            
            # Also get individual predictions for detailed analysis
            predictions = predictor.predict(test_df)
            pred_proba = predictor.predict_proba(test_df)
            
            # Calculate additional metrics manually
            test_labels = test_df[target_column]
            
            from sklearn.metrics import classification_report, confusion_matrix
            
            print(f"\n📊 Detailed Classification Report:")
            print(classification_report(test_labels, predictions))
            
            print(f"\n🔄 Confusion Matrix:")
            cm = confusion_matrix(test_labels, predictions)
            print(cm)
            
        else:
            # Test data doesn't have labels - just predict
            predictions = predictor.predict(test_df)
            pred_proba = predictor.predict_proba(test_df)
            
            print(f"✅ Predictions generated for {len(predictions)} samples")
            print(f"   Prediction distribution:")
            pred_counts = pd.Series(predictions).value_counts()
            for value, count in pred_counts.items():
                percentage = count / len(predictions) * 100
                print(f"     {value}: {count} ({percentage:.1f}%)")
        
        pred_time = time.time() - pred_start
        print(f"⏱️ Prediction time: {format_time(pred_time)}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'predictions': predictions
        })
        
        # Add prediction probabilities
        if pred_proba is not None:
            if hasattr(pred_proba, 'columns'):
                for col in pred_proba.columns:
                    predictions_df[f'prob_{col}'] = pred_proba[col]
            else:
                # Handle numpy array case
                if hasattr(pred_proba, 'ndim') and pred_proba.ndim == 2:
                    for i in range(pred_proba.shape[1]):
                        predictions_df[f'prob_class_{i}'] = pred_proba[:, i]
                elif isinstance(pred_proba, dict):
                    for key, values in pred_proba.items():
                        predictions_df[f'prob_{key}'] = values
        
        # Save results
        predictions_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        predictions_df.to_csv(predictions_file, index=False)
        print(f"💾 Predictions saved to: {predictions_file}")
        
        return predictions, pred_proba
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_feature_importance(predictor, train_df):
    """
    Analyze feature importance for TextPredictor
    
    Parameters
    ----------
    predictor : TextPredictor
        Trained AutoGluon text predictor
    train_df : pd.DataFrame
        Training data
    """
    print("\n" + "="*60)
    print("🎯 Model Analysis")
    print("="*60)
    
    try:
        # TextPredictor doesn't have traditional feature importance like TabularPredictor
        # Instead, we can analyze the model's performance and provide insights
        
        print("📊 Model Summary:")
        print(f"   Model type: TextPredictor")
        print(f"   Training samples: {len(train_df)}")
        
        # Try to get model info if available
        try:
            model_info = predictor.info()
            if model_info:
                print("   Model details:")
                for key, value in model_info.items():
                    if isinstance(value, (str, int, float)):
                        print(f"     {key}: {value}")
        except:
            print("   Model info not available")
        
        # For text models, we can provide general insights
        text_columns = [col for col in train_df.columns 
                       if train_df[col].dtype == 'object' and col != predictor.label]
        
        if text_columns:
            text_col = text_columns[0]
            print(f"\n📝 Text Analysis for column '{text_col}':")
            
            # Basic text statistics
            text_lengths = train_df[text_col].astype(str).str.len()
            word_counts = train_df[text_col].astype(str).str.split().str.len()
            
            print(f"   Character length - Min: {text_lengths.min()}, Max: {text_lengths.max()}, Mean: {text_lengths.mean():.1f}")
            print(f"   Word count - Min: {word_counts.min()}, Max: {word_counts.max()}, Mean: {word_counts.mean():.1f}")
            
            # Vocabulary insights
            all_words = ' '.join(train_df[text_col].astype(str)).lower().split()
            unique_words = len(set(all_words))
            total_words = len(all_words)
            
            print(f"   Vocabulary size: {unique_words} unique words")
            print(f"   Total words: {total_words}")
            print(f"   Vocabulary diversity: {unique_words/total_words:.3f}")
        
        print(f"\n� TextPredictor Insights:")
        print("   - Uses deep learning models (BERT, RoBERTa, etc.) for text understanding")
        print("   - Automatically handles text preprocessing and tokenization")
        print("   - Leverages pre-trained language models for better performance")
        print("   - Feature importance is implicit in the neural network weights")
        
        return None  # TextPredictor doesn't provide traditional feature importance
        
    except Exception as e:
        print(f"⚠️ Could not analyze model: {e}")
        return None

def main():
    """Main execution function"""
    print("🚀 AutoGluon Text Classification Pipeline")
    print("Dataset: MothMalone/data-preprocessing-automl-benchmarks")
    print("="*60)
    
    total_start = time.time()
    
    # Step 1: Load dataset
    train_df, test_df = load_huggingface_dataset("MothMalone/data-preprocessing-automl-benchmarks")
    
    if train_df is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Step 2: Explore dataset
    explore_dataset(train_df, test_df)
    
    # Step 3: Identify target column
    potential_targets = ['label', 'target', 'class', 'y']
    target_col = None
    for col in potential_targets:
        if col in train_df.columns:
            target_col = col
            break
    
    if target_col is None:
        print(f"\n❌ Could not identify target column automatically.")
        print(f"Available columns: {list(train_df.columns)}")
        # Try to use the last column as target
        target_col = train_df.columns[-1]
        print(f"Using last column as target: {target_col}")
    
    # Step 4: Train model
    time_limit = 3600  # 10 minutes - adjust as needed
    predictor = train_autogluon_classifier(train_df, test_df, target_col, time_limit)
    
    if predictor is None:
        print("❌ Training failed. Exiting.")
        return
    
    # Step 5: Evaluate model
    predictions, pred_proba = evaluate_model(predictor, test_df, target_col)
    
    # Step 6: Analyze model
    model_analysis = analyze_feature_importance(predictor, train_df)
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("🎉 Pipeline Complete!")
    print("="*60)
    print(f"Total execution time: {format_time(total_time)}")
    print(f"Model saved in: {predictor.path if predictor else 'N/A'}")
    print("Files generated:")
    print("  - Model files in autogluon_text_model_* directory")
    print("  - predictions_*.csv")
    print("Model type: AutoGluon TextPredictor (Deep Learning)")
    print("="*60)

if __name__ == "__main__":
    main()
