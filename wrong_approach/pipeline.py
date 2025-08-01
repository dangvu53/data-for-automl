import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile, SelectFromModel
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier
from sklearn.decomposition import PCA, FastICA
import os
import joblib
import time
import torch
import wandb
from sentence_transformers import SentenceTransformer
from tpot import TPOTClassifier
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

# --- Global Setup ---
use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'
if use_gpu:
    print("--- GPU detected. Using CUDA for acceleration. ---")
else:
    print("--- No GPU detected. Running on CPU. ---")

print("Loading SentenceTransformer model globally...")
SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("Model loaded.")

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    print("--- Successfully logged into W&B. ---")
except Exception as e:
    print(f"--- Could not log into W&B. Running without logging. Error: {e} ---")
    wandb_api_key = None



def preprocess_clean_data(X):
    """Clean data by removing NaN, Inf values and standardizing features."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Replace NaN values with zeros
    X_cleaned = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float64).max / 10, neginf=np.finfo(np.float64).min / 10)
    
    # Additional defensive check - if we still have NaNs after nan_to_num (rare but possible with certain operations)
    if np.isnan(X_cleaned).any() or np.isinf(X_cleaned).any():
        mask = ~(np.isnan(X_cleaned).any(axis=1) | np.isinf(X_cleaned).any(axis=1))
        if mask.sum() > 0:
            print(f"Warning: Removing {(~mask).sum()} rows with remaining NaN/Inf values")
            X_cleaned = X_cleaned[mask]
        else:
            print("Warning: All rows contain NaN/Inf. Using fallback cleaning.")
            # Fallback: brute-force replacement
            X_cleaned = np.nan_to_num(X_cleaned, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return X_cleaned

def process_text_data(text_series):
    """Basic text preprocessing."""
    if text_series is None:
        return None
        
    # Convert to string in case we have numeric or other types
    clean_texts = text_series.astype(str)
    
    # Basic cleaning
    clean_texts = clean_texts.str.lower()
    # Remove excessive whitespace
    clean_texts = clean_texts.str.replace(r'\s+', ' ', regex=True)
    # Strip leading and trailing whitespace
    clean_texts = clean_texts.str.strip()
    
    # Remove empty strings or replace with placeholder
    clean_texts = clean_texts.replace('', 'empty_text')
    
    return clean_texts

def process_label_data(labels):
    """Clean and standardize label data."""
    if isinstance(labels, pd.Series):
        # Handle missing values
        if labels.isna().any():
            print(f"Warning: Found {labels.isna().sum()} missing labels. Dropping these entries.")
            labels = labels.dropna()
        
        # Convert to numeric if possible
        if pd.api.types.is_object_dtype(labels):
            try:
                labels = pd.to_numeric(labels, errors='coerce')
                # Drop any that couldn't be converted
                if labels.isna().any():
                    print(f"Warning: Found {labels.isna().sum()} labels that couldn't be converted to numeric.")
                    labels = labels.dropna()
            except:
                print("Labels could not be converted to numeric, using as-is.")
        
        # Ensure integer labels
        if pd.api.types.is_numeric_dtype(labels):
            if not labels.equals(labels.astype(int)):
                print("Converting non-integer labels to integers.")
                # Map to consecutive integers starting from 0
                unique_labels = labels.unique()
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                labels = labels.map(label_map)
    
    return labels

# Modify the encode_texts function to ensure clean inputs
def encode_texts(texts):
    """Encode texts with error handling and preprocessing."""
    if texts is None or len(texts) == 0:
        print("Warning: No texts provided for encoding.")
        return np.array([])
        
    if isinstance(texts, pd.Series):
        # Apply basic text cleaning
        texts = process_text_data(texts)
        texts = texts.tolist()
    
    try:
        embeddings = SBERT_MODEL.encode(texts, show_progress_bar=True, device=device)
        
        # Verify embeddings quality
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            print("Warning: NaN/Inf values detected in embeddings. Cleaning...")
            embeddings = preprocess_clean_data(embeddings)
            
        return embeddings
        
    except Exception as e:
        print(f"Error during text encoding: {e}")
        # Fallback: return zero embeddings
        embedding_size = SBERT_MODEL.get_sentence_embedding_dimension()
        return np.zeros((len(texts), embedding_size))


class MetaFeatureExtractor:
    def characterize(self, train_df, text_column, label_column, sample_embeddings_df):
        print("Phase 1: Characterizing dataset with advanced, model-based metrics...")
        
        sample_embeddings = sample_embeddings_df.values
        num_rows, num_cols = train_df.shape
        class_imbalance = train_df[label_column].value_counts(normalize=True).min()
        iso_forest = IsolationForest(contamination= 0.05, random_state=42)
    
        outlier_preds = iso_forest.fit_predict(sample_embeddings)
        outlier_fraction = np.mean(outlier_preds < -0.5)
        temp_df = sample_embeddings_df.copy()
        
        temp_df['label'] = train_df.loc[sample_embeddings_df.index, label_column].values
        centroids = temp_df.groupby('label').mean()
        noise_estimate = 0
        if len(centroids) > 1:
            within_class_variances = temp_df.groupby('label').var(numeric_only=True).mean().mean()
            centroid_distances = pairwise_distances(centroids).mean()
            noise_estimate = within_class_variances / (centroid_distances + 1e-6)
        meta_features = {'num_rows': num_rows, 'class_imbalance': class_imbalance, 'outlier_fraction': outlier_fraction, 'noise_estimate': noise_estimate}
        print(f"Meta-features extracted: {meta_features}")
        return meta_features

class MetaStrategyPredictor:
    def predict_policy(self, meta_features):
        print("Phase 2: Predicting preprocessing policy with nuanced heuristics...")
        policy = {'priority': [], 'budget': [], 'class_weight': None, 'use_smote': False, 'handle_outliers': False}
        
        if meta_features['class_imbalance'] < 0.2:
            policy['class_weight'] = 'balanced'
            if meta_features['num_rows'] >= 1000:
                policy['use_smote'] = True
                print("Policy decision: Handle imbalance with SMOTE and balanced class weights.")
            else:
                print("Policy decision: Handle imbalance with balanced class weights only (dataset too small for SMOTE).")
                
        if meta_features['outlier_fraction'] > 0.05:
            policy['handle_outliers'] = True
            print("Policy decision: Outliers detected, will prioritize robust transformations.")
            
        if meta_features['noise_estimate'] > 0.5:
            policy['priority'] = ['transformation', 'selection']
            policy['budget'] = [0.7, 0.3]
            print("Policy decision: High noise detected, prioritizing transformation.")
            
        else:
            policy['priority'] = ['selection', 'transformation']
            policy['budget'] = [0.6, 0.4]
            print("Policy decision: Using balanced approach for selection and transformation.")
        return policy

def preprocess_clean_data(X):
    """Clean data by replacing NaN and Inf values."""
    # Replace infinities with large values
    X_cleaned = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return X_cleaned


class GuidedEvolutionaryOptimizer:
    def __init__(self, policy):
        self.policy = policy

    def find_best_pipeline(self, X_train_embedded, y_train, cache_path):
        if os.path.exists(cache_path):
            print(f"Loading cached TPOT pipeline from {cache_path}")
            return joblib.load(cache_path)
        print("Phase 3: Starting guided evolutionary pipeline optimization...")
        operator_categories = {
            'selection': {
                'sklearn.feature_selection.SelectFwe': {
                    'alpha': [0.001, 0.005, 0.01, 0.05, 0.1],
                    'score_func': {'sklearn.feature_selection.f_classif'}
                },
                'sklearn.feature_selection.SelectPercentile': {'percentile': range(10, 100, 10), 'score_func': {'sklearn.feature_selection.f_classif'}},
                'sklearn.feature_selection.SelectFromModel': {'estimator': {'sklearn.ensemble.ExtraTreesClassifier': {'n_estimators': [100], 'criterion': ['gini', 'entropy'], 'random_state': [42]}}, 'threshold': np.arange(0, 1.01, 0.05).tolist()},
            },
            'transformation': {
                 'sklearn.preprocessing.StandardScaler': {}, 'sklearn.preprocessing.MinMaxScaler': {},
                 'sklearn.preprocessing.Normalizer': {'norm': ['l1', 'l2']},
                 'sklearn.decomposition.PCA': {'n_components': [0.75, 0.95]},
                 'sklearn.decomposition.FastICA': {'tol': np.arange(0.0, 1.01, 0.05).tolist()}
            }
        }
        tpot_config = {}
        for category in self.policy.get('priority', []):
            if category in operator_categories:
                tpot_config.update(operator_categories[category])
        if self.policy.get('handle_outliers'):
            tpot_config['sklearn.preprocessing.RobustScaler'] = {}
            print("Added RobustScaler to handle outliers.")
        tpot_config.update({
            'sklearn.linear_model.LogisticRegression': {'penalty': ["l2"], 'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 'class_weight': [self.policy.get('class_weight')], 'max_iter': [2000]},
            'sklearn.ensemble.RandomForestClassifier': {  # Using RandomForest instead of single decision tree
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'], 
                'max_depth': range(2, 11), 
                'class_weight': [self.policy.get('class_weight')],
                'random_state': [42]
            },
        })
        population_size, generations = 60, 8
        print(f"Configuring TPOT with population_size={population_size}, generations={generations}...")
        tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2, random_state=42, scoring='accuracy', cv=3, config_dict=tpot_config, n_jobs=-1)
        tpot.fit(X_train_embedded, y_train)
        best_pipeline = tpot.fitted_pipeline_
        joblib.dump(best_pipeline, cache_path)
        return best_pipeline

        # Modify create_data_subset to include proper preprocessing
def create_data_subset(dataset_name, challenge_type, max_train_samples=2000, severity=0.8):
    print(f"\nLoading/Creating full dataset for '{dataset_name}' to create '{challenge_type}' subset...")
    np.random.seed(42)
    
    # Load datasets with error handling
    try:
        if dataset_name == 'casehold':
            full_dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "casehold")
            train_df_full, val_df, test_df = (full_dataset['train'].to_pandas(), 
                                             full_dataset['validation'].to_pandas(), 
                                             full_dataset['test'].to_pandas())
                                             
            for df in [train_df_full, val_df, test_df]: 
                # Clean and combine columns first
                text_columns = ['citing_prompt', 'holding_0', 'holding_1', 'holding_2', 'holding_3', 'holding_4']
                for col in text_columns:
                    df[col] = df[col].fillna('').astype(str)
                
                df['text'] = df['citing_prompt'] + ' [SEP] ' + df['holding_0'] + ' [SEP] ' + \
                             df['holding_1'] + ' [SEP] ' + df['holding_2'] + ' [SEP] ' + \
                             df['holding_3'] + ' [SEP] ' + df['holding_4']
                             
                # Process the combined text
                df['text'] = process_text_data(df['text'])
                
            label_column = 'label'
            
        elif dataset_name == 'anli_r1':
            full_dataset = load_dataset("facebook/anli")
            train_df_full, val_df, test_df = (full_dataset['train_r1'].to_pandas(), 
                                             full_dataset['dev_r1'].to_pandas(), 
                                             full_dataset['test_r1'].to_pandas())
                                             
            for df in [train_df_full, val_df, test_df]:
                df['premise'] = df['premise'].fillna('').astype(str)
                df['hypothesis'] = df['hypothesis'].fillna('').astype(str)
                df['text'] = df['premise'] + " [SEP] " + df['hypothesis']
                df['text'] = process_text_data(df['text'])
                
            label_column = 'label'
            
        elif dataset_name == 'scienceqa':
            full_dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "scienceqa")
            train_df_full, val_df, test_df = (full_dataset['train'].to_pandas(), 
                                             full_dataset['validation'].to_pandas(), 
                                             full_dataset['test'].to_pandas())
                                             
            for df in [train_df_full, val_df, test_df]:
                # Process choices array
                choices_str = ""
                choices_list = df.get('choices', [])
                if isinstance(choices_list, list) and len(choices_list) > 0:
                    if isinstance(choices_list[0], list):
                        choices_str = " | ".join([str(c) for c in choices_list[0]])
                
                # Clean and combine fields
                fields = {
                    'subject': 'SUBJECT: ', 
                    'topic': 'TOPIC: ', 
                    'question': '[QUESTION] ',
                    'hint': '[HINT] ',
                    'lecture': '[LECTURE] '
                }
                
                text_parts = []
                for field, prefix in fields.items():
                    value = df.get(field, '')
                    if value is not None:
                        text_parts.append(f"{prefix}{value}")
                
                text_parts.append(f"[CHOICES] {choices_str}")
                df['text'] = ' '.join(text_parts)
                df['text'] = process_text_data(df['text'])
                
            label_column = 'answer'
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        raise
    
    # Process labels for all dataframes
    for df in [train_df_full, val_df, test_df]:
        # Clean and standardize labels
        df[label_column] = process_label_data(df[label_column])
        
        # Drop rows with missing values in critical columns
        initial_len = len(df)
        df.dropna(subset=[label_column, 'text'], inplace=True)
        dropped_rows = initial_len - len(df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing label or text values")
        
        # Ensure label is integer
        df[label_column] = df[label_column].astype(int)
    
    # Create challenge-specific subsets with clean data
    if challenge_type == 'imbalance':
        print(f"Creating imbalanced subset of size {max_train_samples}...")
        majority_class = train_df_full[label_column].mode()[0]
        n_majority_target = int(max_train_samples * severity)
        
        majority_df_full = train_df_full[train_df_full[label_column] == majority_class]
        minority_df_full = train_df_full[train_df_full[label_column] != majority_class]
        
        n_majority_actual = min(n_majority_target, len(majority_df_full))
        majority_df_sampled = majority_df_full.sample(n=n_majority_actual, random_state=42, replace=False)
        
        n_minority_needed = max_train_samples - n_majority_actual
        minority_df_sampled = minority_df_full.sample(n=min(n_minority_needed, len(minority_df_full)), 
                                                    random_state=42, 
                                                    replace=(n_minority_needed > len(minority_df_full)))
        
        train_df = pd.concat([majority_df_sampled, minority_df_sampled]).sample(frac=1, random_state=42)
        
    else:
        # Ensure we don't sample more than available
        max_train_samples = min(max_train_samples, len(train_df_full))
        train_df = train_df_full.sample(n=max_train_samples, random_state=42)
        
        if challenge_type == 'noise':
            print(f"Creating noisy label subset...")
            n_noisy = int(len(train_df) * severity * 0.4)
            noisy_indices = np.random.choice(train_df.index, size=n_noisy, replace=False)
            unique_labels = train_df[label_column].unique()
            
            if len(unique_labels) > 1:
                for idx in noisy_indices:
                    current_label = train_df.loc[idx, label_column]
                    other_labels = [l for l in unique_labels if l != current_label]
                    if other_labels: 
                        train_df.loc[idx, label_column] = np.random.choice(other_labels)
            else:
                print("Warning: Cannot create noisy labels - only one class present")
                    
        elif challenge_type == 'outliers':
            print(f"Creating outlier subset...")
            try:
                embeddings = encode_texts(train_df['text'])
                
                # Handle potential failures in embedding generation
                if len(embeddings) == 0 or np.isnan(embeddings).any() or np.isinf(embeddings).any():
                    print("Warning: Issues with embeddings. Cleaning before outlier detection.")
                    embeddings = preprocess_clean_data(embeddings)
                
                # Calculate outliers based on distance from centroid
                centroid = embeddings.mean(axis=0)
                distances = np.linalg.norm(embeddings - centroid, axis=1)
                
                # Get indices of normal and outlier samples
                n_outliers = int(len(train_df) * severity * 0.15)
                outlier_indices = np.argsort(distances)[-n_outliers:]
                normal_indices = np.argsort(distances)[:(len(train_df) - n_outliers)]
                
                # Reconstruct dataframe with outliers
                train_df = train_df.iloc[np.concatenate([normal_indices, outlier_indices])]
            except Exception as e:
                print(f"Error creating outlier subset: {e}. Using original sample instead.")
    
    # Verify and report final sizes
    print(f"Final train size: {len(train_df)}. Val size: {len(val_df)}. Test size: {len(test_df)}")
    
    # Verify all data frames have the required columns and no missing values
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if label_column not in df.columns or 'text' not in df.columns:
            missing_cols = [col for col in [label_column, 'text'] if col not in df.columns]
            raise ValueError(f"{name} dataframe is missing columns: {missing_cols}")
            
        if df[label_column].isna().any() or df['text'].isna().any():
            print(f"Warning: {name} dataframe has {df[label_column].isna().sum()} missing labels and {df['text'].isna().sum()} missing texts")
            # Fill any remaining NaN values
            df[label_column] = df[label_column].fillna(df[label_column].mode()[0])
            df['text'] = df['text'].fillna('missing_text')
    
    return train_df, val_df, test_df

def analyze_and_visualize(name, challenge_type, before_embeddings, before_labels, after_embeddings, after_labels, output_base_dir):
    # This function remains the same as the last corrected version.
    print(f"--- Generating visualizations for {name} ---")
    plot_paths = {}
    dist_plot_path = os.path.join(output_base_dir, f"{name}_distribution_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    sns.countplot(x=before_labels, ax=ax1, palette="viridis")
    ax1.set_title("Distribution Before Preprocessing", fontsize=16)
    ax1.set_xlabel("Class Label")
    ax1.set_ylabel("Count")
    sns.countplot(x=after_labels, ax=ax2, palette="magma")
    ax2.set_title("Distribution After Preprocessing (SMOTE)", fontsize=16)
    ax2.set_xlabel("Class Label")
    ax2.set_ylabel("Count")
    fig.suptitle(f"Class Distribution Comparison for {name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(dist_plot_path)
    plt.close()
    plot_paths['distribution_comparison'] = dist_plot_path
    umap_plot_path = os.path.join(output_base_dir, f"{name}_umap_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    reducer = umap.UMAP(random_state=42)
    before_2d = reducer.fit_transform(before_embeddings)
    sns.scatterplot(x=before_2d[:, 0], y=before_2d[:, 1], hue=before_labels, palette="viridis", ax=ax1, s=50, alpha=0.7)
    ax1.set_title("Before Preprocessing", fontsize=16)
    ax1.grid(True)
    if challenge_type == 'outliers':
        centroid = before_embeddings.mean(axis=0)
        distances = np.linalg.norm(before_embeddings - centroid, axis=1)
        n_outliers = int(len(before_embeddings) * 0.8 * 0.15)
        outlier_indices_2d = np.argsort(distances)[-n_outliers:]
        ax1.scatter(before_2d[outlier_indices_2d, 0], before_2d[outlier_indices_2d, 1], s=150, facecolors='none', edgecolors='r', linewidth=2, label='Identified Outliers')
    if challenge_type == 'noise' or challenge_type == 'imbalance':
        temp_df = pd.DataFrame(before_embeddings)
        temp_df['label'] = before_labels
        centroids_2d = reducer.transform(temp_df.groupby('label').mean().values)
        ax1.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=200, marker='X', c='black', label='Class Centroids')
    ax1.legend()
    if after_embeddings.shape[1] >= 2:
        after_2d = umap.UMAP(random_state=42).fit_transform(after_embeddings)
        sns.scatterplot(x=after_2d[:, 0], y=after_2d[:, 1], hue=after_labels, palette="viridis", ax=ax2, s=50, alpha=0.7)
        ax2.set_title("After Optimized Preprocessing", fontsize=16)
        ax2.grid(True)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Processed data has < 2 dimensions', horizontalalignment='center', verticalalignment='center')
    fig.suptitle(f"UMAP Visualization for {name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(umap_plot_path)
    plt.close()
    plot_paths['umap_comparison'] = umap_plot_path
    print(f"Visualizations saved.")
    return plot_paths


def identity_function(x):
    """Identity function that returns input unchanged."""
    return x

# --- Main Execution Block with ENHANCED CACHING and FIX ---
def loader_casehold_imbalanced():
    return create_data_subset('casehold', 'imbalance', severity=0.95)

def loader_anli_noisy():
    return create_data_subset('anli_r1', 'noise', severity=0.7)

def loader_scienceqa_outliers():
    return create_data_subset('scienceqa', 'outliers', severity=0.8)

def main():
    if not wandb_api_key:
        print("W&B API key not available. Skipping experiment execution.")
        return
        
    datasets_to_process = {
        'casehold_imbalanced': {'loader': loader_casehold_imbalanced, 'label_column': 'label', 'challenge': 'imbalance'},
        'anli_r1_noisy': {'loader': loader_anli_noisy, 'label_column': 'label', 'challenge': 'noise'},
        'scienceqa_outliers': {'loader': loader_scienceqa_outliers, 'label_column': 'answer', 'challenge': 'outliers'},
    }
    
    output_base_dir = '/kaggle/working/cleaned_datasets_experiment'
    os.makedirs(output_base_dir, exist_ok=True)

    for name, config in datasets_to_process.items():
        run_name = f"exp_{name}_{int(time.time())}"
        run = wandb.init(project="meta-learning-preprocessing", name=run_name, job_type="eval")
        print(f"\n{'='*25}\nRUNNING EXPERIMENT: {name}\n{'='*25}")
        
        dataset_cache_dir = os.path.join(output_base_dir, name, 'cache')
        os.makedirs(dataset_cache_dir, exist_ok=True)
        
        train_df_cache = os.path.join(dataset_cache_dir, 'train_subset.csv')
        val_df_cache = os.path.join(dataset_cache_dir, 'val_subset.csv')
        test_df_cache = os.path.join(dataset_cache_dir, 'test_subset.csv')
        
        embeddings_cache_path = os.path.join(dataset_cache_dir, 'embeddings.joblib')
        tpot_pipeline_cache_path = os.path.join(dataset_cache_dir, 'tpot_pipeline.joblib')
        

        if os.path.exists(train_df_cache) and os.path.exists(val_df_cache) and os.path.exists(test_df_cache):
            print(f"Loading cached data subset for {name}...")
            train_df = pd.read_csv(train_df_cache)
            val_df = pd.read_csv(val_df_cache)
            test_df = pd.read_csv(test_df_cache)
        else:
            print(f"Generating new data subset for {name}...")
            train_df, val_df, test_df = config['loader']()
            train_df.to_csv(train_df_cache, index=False)
            val_df.to_csv(val_df_cache, index=False)
            test_df.to_csv(test_df_cache, index=False)
            print(f"Saved data subset to cache.")
            
        text_column, label_column = 'text', config['label_column']
        
        if os.path.exists(embeddings_cache_path):
            print("Loading cached embeddings...")
            X_train_embedded, X_val_embedded, X_test_embedded = joblib.load(embeddings_cache_path)
        else:
            # Check if validation or test embeddings are None and regenerate them
            if X_val_embedded is None:
                print("Validation embeddings are None, generating them...")
                X_val_embedded = encode_texts(val_df[text_column])
                print(f"Generated validation embeddings with shape: {X_val_embedded.shape}")
            
            if X_test_embedded is None:
                print("Test embeddings are None, generating them...")
                X_test_embedded = encode_texts(test_df[text_column])
                print(f"Generated test embeddings with shape: {X_test_embedded.shape}")
            
            # Save fixed embeddings back to cache
            joblib.dump((X_train_embedded, X_val_embedded, X_test_embedded), embeddings_cache_path)
            print("Saved fixed embeddings to cache.")
            print("Generating and caching embeddings for all splits...")
            X_train_embedded = encode_texts(train_df[text_column])
            X_val_embedded = encode_texts(val_df[text_column])
            X_test_embedded = encode_texts(test_df[text_column])
            joblib.dump((X_train_embedded, X_val_embedded, X_test_embedded), embeddings_cache_path)
            
        X_train_embedded_df = pd.DataFrame(X_train_embedded, index=train_df.index)
        
        meta_extractor = MetaFeatureExtractor()
        sample_indices = np.random.choice(X_train_embedded_df.index, size=min(500, len(X_train_embedded_df)), replace=False)
        meta_features = meta_extractor.characterize(train_df, text_column, label_column, X_train_embedded_df.loc[sample_indices])
        
        meta_predictor = MetaStrategyPredictor()
        policy = meta_predictor.predict_policy(meta_features)
        
        optimizer = GuidedEvolutionaryOptimizer(policy)
        best_pipeline_structure = optimizer.find_best_pipeline(X_train_embedded, train_df[label_column], tpot_pipeline_cache_path)
        
        if isinstance(best_pipeline_structure, Pipeline):
            transformer_steps = best_pipeline_structure.steps[:-1]
            # Use our defined identity_function instead of lambda
            final_preprocessor = Pipeline(transformer_steps) if transformer_steps else FunctionTransformer(identity_function, validate=False)
        else:
            # Use our defined identity_function instead of lambda
            final_preprocessor = FunctionTransformer(identity_function, validate=False)
        
        # Ensure no NaN/Inf values in embeddings before fitting preprocessor
        X_train_embedded_clean = preprocess_clean_data(X_train_embedded)
        
        try:
            final_preprocessor.fit(X_train_embedded_clean, train_df[label_column])
            processed_train_data = final_preprocessor.transform(X_train_embedded_clean)
            
            # Clean any NaN/Inf values that might have been introduced
            processed_train_data = preprocess_clean_data(processed_train_data)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            print("Using identity transformation as fallback")
            final_preprocessor = FunctionTransformer(identity_function, validate=False)
            processed_train_data = X_train_embedded_clean
        
        # --- FIX: Create and save the "after" datasets ---
        processed_train_data = final_preprocessor.transform(X_train_embedded)
        
        # Don't transform validation and test data - keep original embeddings
        # This prevents corruption of validation and test data
        processed_val_data = X_val_embedded
        processed_test_data = X_test_embedded
        
        before_train_labels = train_df[label_column].values
        after_train_labels = before_train_labels.copy()

        if policy.get('use_smote'):
            print("Applying SMOTE post-transformation...")
            unique_classes, class_counts = np.unique(after_train_labels, return_counts=True)
            min_class_size = class_counts.min()

            if min_class_size > 1:
                k_neighbors = max(1, min_class_size - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                processed_train_data, after_train_labels = smote.fit_resample(processed_train_data, after_train_labels)
                print(f"Data resampled with SMOTE. New train size: {len(processed_train_data)}")
            else:
                print(f"Skipping SMOTE: Smallest class has only {min_class_size} sample(s).")

        # Create and save the final "after" dataframes
        feature_names = [f'feature_{i}' for i in range(processed_train_data.shape[1])]
        
        after_train_df = pd.DataFrame(processed_train_data, columns=feature_names)
        after_train_df[label_column] = after_train_labels
        
        after_val_df = pd.DataFrame(processed_val_data, columns=feature_names)
        after_val_df[label_column] = val_df[label_column].values
        
        after_test_df = pd.DataFrame(processed_test_data, columns=feature_names)
        after_test_df[label_column] = test_df[label_column].values

        after_train_df.to_csv(os.path.join(dataset_cache_dir, 'after_train.csv'), index=False)
        after_val_df.to_csv(os.path.join(dataset_cache_dir, 'after_val.csv'), index=False)
        after_test_df.to_csv(os.path.join(dataset_cache_dir, 'after_test.csv'), index=False)
        print("Saved 'after' datasets to cache.")

        
        wandb.config.update({"dataset_name": name, "challenge_type": name.split('_')[-1], "train_samples": len(train_df), **meta_features, "policy": policy})
        wandb.log({"best_pipeline_str": str(best_pipeline_structure)})
        
        plot_paths = analyze_and_visualize(
            name=name, challenge_type=config['challenge'],
            before_embeddings=X_train_embedded, before_labels=before_train_labels,
            after_embeddings=processed_train_data, after_labels=after_train_labels,
            output_base_dir=output_base_dir
        )
        
        log_data = {"Distribution_Comparison": wandb.Image(plot_paths['distribution_comparison']), "UMAP_Comparison": wandb.Image(plot_paths['umap_comparison'])}
        if wandb_api_key and run: wandb.log(log_data)
        
        artifact = wandb.Artifact(name=f"preprocessor_{name}", type="preprocessing_pipeline")
        pipeline_path = os.path.join(output_base_dir, f"{name}_pipeline.joblib")
        joblib.dump(final_preprocessor, pipeline_path)
        artifact.add_file(pipeline_path)
        if wandb_api_key and run: run.log_artifact(artifact)
        
        print(f"--- Finished experiment for {name}. Check W&B run for results. ---")
        if wandb_api_key and run: 
            wandb.finish()

main()


