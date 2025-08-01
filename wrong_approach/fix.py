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


def identity_function(x):
    """Identity function that returns input unchanged."""
    return x


def encode_texts(texts):
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    return SBERT_MODEL.encode(texts, show_progress_bar=True, device=device)


def preprocess_clean_data(X):
    """Remove NaN and Inf values from data."""
    if isinstance(X, pd.DataFrame):
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        X_clean = np.copy(X)
        X_clean[~np.isfinite(X_clean)] = 0
        return X_clean


class MetaFeatureExtractor:
    def characterize(self, train_df, text_column, label_column, sample_embeddings_df):
        print("Phase 1: Characterizing dataset with advanced, model-based metrics...")
        
        sample_embeddings = sample_embeddings_df.values
        num_rows, num_cols = train_df.shape
        
        # More detailed class distribution analysis
        class_counts = train_df[label_column].value_counts()
        class_imbalance = class_counts.min() / class_counts.max()
        
        # Count classes with very few samples (potential rare classes vs outliers)
        total_samples = len(train_df)
        rare_class_threshold = 0.01  # Classes with less than 1% of samples
        rare_classes = class_counts[class_counts/total_samples < rare_class_threshold]
        num_rare_classes = len(rare_classes)
        
        # Check if this is ScienceQA dataset with label 4 issue
        is_scienceqa = 'answer' in train_df.columns and 4 in train_df[label_column].unique()
        has_scienceqa_label4_issue = is_scienceqa and (train_df[label_column] == 4).mean() < 0.05
        
        # Run Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_preds = iso_forest.fit_predict(sample_embeddings)
        outlier_fraction = np.mean(outlier_preds == -1)
        
        # Enhanced analysis - check if outliers are concentrated in specific classes
        temp_df = sample_embeddings_df.copy()
        temp_df['label'] = train_df.loc[sample_embeddings_df.index, label_column].values
        temp_df['outlier'] = outlier_preds == -1
        
        # Calculate outlier concentration by class
        outliers_by_class = temp_df.groupby('label')['outlier'].mean()
        max_outlier_concentration = outliers_by_class.max() if not outliers_by_class.empty else 0
        
        # Calculate within-class cohesion and between-class separation
        within_class_variances = []
        for label, group in temp_df.groupby('label'):
            if len(group) >= 2:  # Need at least 2 samples to calculate variance
                group_embeddings = group.drop(['label', 'outlier'], axis=1, errors='ignore').values
                within_class_variances.append(np.mean(np.var(group_embeddings, axis=0)))
        
        mean_within_variance = np.mean(within_class_variances) if within_class_variances else 0
        
        # Between-class distances
        centroids = temp_df.groupby('label').mean().drop(['outlier'], axis=1, errors='ignore')
        
        between_class_distances = []
        if len(centroids) > 1:
            centroid_values = centroids.values
            for i in range(len(centroid_values)):
                for j in range(i+1, len(centroid_values)):
                    between_class_distances.append(np.linalg.norm(centroid_values[i] - centroid_values[j]))
            
            mean_between_distance = np.mean(between_class_distances) if between_class_distances else 0
            separation_ratio = mean_between_distance / (mean_within_variance + 1e-6)
        else:
            mean_between_distance = 0
            separation_ratio = 0
        
        # Measure of noise - check how intermingled different classes are
        noise_estimate = 0
        if len(centroids) > 1:
            # Calculate how much samples deviate from their class centroids relative to inter-class distances
            noise_estimate = mean_within_variance / (mean_between_distance + 1e-6)
        
        meta_features = {
            'num_rows': num_rows,
            'class_imbalance': class_imbalance,
            'outlier_fraction': outlier_fraction,
            'noise_estimate': noise_estimate,
            'num_rare_classes': num_rare_classes,
            'max_outlier_concentration': max_outlier_concentration,
            'separation_ratio': separation_ratio,
            'is_scienceqa_with_label4_issue': has_scienceqa_label4_issue
        }
        
        print(f"Meta-features extracted: {meta_features}")
        return meta_features


class MetaStrategyPredictor:
    def predict_policy(self, meta_features):
        print("Phase 2: Predicting preprocessing policy with nuanced heuristics...")
        policy = {
            'priority': [], 
            'budget': [], 
            'class_weight': None, 
            'use_smote': False, 
            'handle_outliers': False,
            'drop_outliers': False,
            'scienceqa_special_handling': False
        }
        
        # Special handling for ScienceQA dataset with label 4 issue
        if meta_features.get('is_scienceqa_with_label4_issue', False):
            policy['scienceqa_special_handling'] = True
            policy['class_weight'] = 'balanced'
            policy['use_smote'] = False  # Don't use SMOTE for ScienceQA with rare label 4
            print("Policy decision: Detected ScienceQA with rare label 4. Using special handling with balanced class weights.")
            
        # General class imbalance handling    
        elif meta_features['class_imbalance'] < 0.2:
            policy['class_weight'] = 'balanced'
            if meta_features['num_rows'] >= 1000:
                # Only use SMOTE if we don't have outlier concentration issues
                if meta_features['max_outlier_concentration'] < 0.3:
                    policy['use_smote'] = True
                    print("Policy decision: Handle imbalance with SMOTE and balanced class weights.")
                else:
                    print("Policy decision: Using balanced class weights only (high outlier concentration detected).")
            else:
                print("Policy decision: Handle imbalance with balanced class weights only (dataset too small for SMOTE).")
        
        # Outlier handling - separate from class imbalance
        if meta_features['outlier_fraction'] > 0.05:
            if meta_features['max_outlier_concentration'] > 0.5 and meta_features['separation_ratio'] < 1.0:
                # High concentration of outliers in specific classes with poor separation
                policy['drop_outliers'] = True
                print("Policy decision: High concentration of outliers detected. Will drop outlier samples.")
            else:
                # General outliers spread across classes
                policy['handle_outliers'] = True
                print("Policy decision: Outliers detected, will prioritize robust transformations.")
            
        # Noise handling    
        if meta_features['noise_estimate'] > 0.5:
            policy['priority'] = ['transformation', 'selection']
            policy['budget'] = [0.7, 0.3]
            print("Policy decision: High noise detected, prioritizing transformation.")
        else:
            policy['priority'] = ['selection', 'transformation']
            policy['budget'] = [0.6, 0.4]
            print("Policy decision: Using balanced approach for selection and transformation.")
            
        return policy


class GuidedEvolutionaryOptimizer:
    def __init__(self, policy):
        self.policy = policy

    def find_best_pipeline(self, X_train_embedded, y_train, cache_path):
        if os.path.exists(cache_path):
            print(f"Loading cached TPOT pipeline from {cache_path}")
            return joblib.load(cache_path)
        
        print("Phase 3: Starting guided evolutionary pipeline optimization...")
        
        # Base operator categories
        operator_categories = {
            'selection': {
                'sklearn.feature_selection.SelectFwe': {
                    'alpha': [0.001, 0.005, 0.01, 0.05, 0.1],
                    'score_func': {'sklearn.feature_selection.f_classif'}
                },
                'sklearn.feature_selection.SelectPercentile': {
                    'percentile': range(10, 100, 10), 
                    'score_func': {'sklearn.feature_selection.f_classif'}
                },
                'sklearn.feature_selection.SelectFromModel': {
                    'estimator': {
                        'sklearn.ensemble.ExtraTreesClassifier': {
                            'n_estimators': [100], 
                            'criterion': ['gini', 'entropy'], 
                            'random_state': [42]
                        }
                    }, 
                    'threshold': np.arange(0, 1.01, 0.05).tolist()
                },
            },
            'transformation': {
                'sklearn.preprocessing.StandardScaler': {}, 
                'sklearn.preprocessing.MinMaxScaler': {},
                'sklearn.preprocessing.Normalizer': {'norm': ['l1', 'l2']},
                'sklearn.decomposition.PCA': {'n_components': [0.75, 0.95]},
                'sklearn.decomposition.FastICA': {'tol': np.arange(0.0, 1.01, 0.05).tolist()}
            }
        }
        
        # Add additional operators based on policy
        tpot_config = {}
        for category in self.policy.get('priority', []):
            if category in operator_categories:
                tpot_config.update(operator_categories[category])
        
        # Add robust scaler if handling outliers
        if self.policy.get('handle_outliers'):
            tpot_config['sklearn.preprocessing.RobustScaler'] = {}
            print("Added RobustScaler to handle outliers.")
            
        # Add PowerTransformer for ScienceQA special handling
        if self.policy.get('scienceqa_special_handling'):
            tpot_config['sklearn.preprocessing.PowerTransformer'] = {'method': ['yeo-johnson'], 'standardize': [True, False]}
            print("Added PowerTransformer for ScienceQA special handling.")
        
        # Add classification estimators
        class_weight = self.policy.get('class_weight')
        tpot_config.update({
            'sklearn.linear_model.LogisticRegression': {
                'penalty': ["l1", "l2"], 
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 
                'class_weight': [class_weight], 
                'max_iter': [2000]
            },
            'sklearn.tree.DecisionTreeClassifier': {
                'n_estimators': [100], 
                'criterion': ['gini', 'entropy'], 
                'max_depth': range(2, 11), 
                'class_weight': [class_weight]
            },
        })
        
        # Adjust TPOT parameters based on dataset size
        if X_train_embedded.shape[0] > 5000:
            population_size, generations = 80, 10
        else:
            population_size, generations = 60, 8
            
        print(f"Configuring TPOT with population_size={population_size}, generations={generations}...")
        
        tpot = TPOTClassifier(
            generations=generations, 
            population_size=population_size,
            verbosity=2,
            random_state=42,
            scoring='balanced_accuracy' if self.policy.get('class_weight') else 'accuracy',
            cv=3,
            config_dict=tpot_config,
            n_jobs=-1
        )
        
        tpot.fit(X_train_embedded, y_train)
        best_pipeline = tpot.fitted_pipeline_
        joblib.dump(best_pipeline, cache_path)
        return best_pipeline


def create_data_subset(dataset_name, challenge_type, max_train_samples=2000, severity=0.8):
    print(f"\nLoading/Creating full dataset for '{dataset_name}' to create '{challenge_type}' subset...")
    np.random.seed(42)
    if dataset_name == 'casehold':
        full_dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "casehold")
        train_df_full, val_df, test_df = (full_dataset['train'].to_pandas(), full_dataset['validation'].to_pandas(), full_dataset['test'].to_pandas())
        for df in [train_df_full, val_df, test_df]: df['text'] = (df['citing_prompt'].fillna('') + ' [SEP] ' + df['holding_0'].fillna('') + ' [SEP] ' + df['holding_1'].fillna('') + ' [SEP] ' + df['holding_2'].fillna('') + ' [SEP] ' + df['holding_3'].fillna('') + ' [SEP] ' + df['holding_4'].fillna(''))
        label_column = 'label'
    elif dataset_name == 'anli_r1':
        full_dataset = load_dataset("facebook/anli")
        train_df_full, val_df, test_df = (full_dataset['train_r1'].to_pandas(), full_dataset['dev_r1'].to_pandas(), full_dataset['test_r1'].to_pandas())
        for df in [train_df_full, val_df, test_df]: df['text'] = df['premise'].fillna('') + " [SEP] " + df['hypothesis'].fillna('')
        label_column = 'label'
    elif dataset_name == 'scienceqa':
        full_dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "scienceqa")
        train_df_full, val_df, test_df = (full_dataset['train'].to_pandas(), full_dataset['validation'].to_pandas(), full_dataset['test'].to_pandas())
        for df in [train_df_full, val_df, test_df]:
            choices_list = df.get('choices', [])
            if isinstance(choices_list, list) and len(choices_list) > 0 and isinstance(choices_list[0], list):
                 choices_str = " | ".join(choices_list[0])
            else:
                 choices_str = ""
            df['text'] = ('SUBJECT: ' + str(df.get('subject', '')) + ' TOPIC: ' + str(df.get('topic', '')) + ' [QUESTION] ' + str(df.get('question', '')) + ' [CHOICES] ' + choices_str + ' [HINT] ' + str(df.get('hint', '')) + ' [LECTURE] ' + str(df.get('lecture', '')))
        label_column = 'answer'
    for df in [train_df_full, val_df, test_df]:
        df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
        df.dropna(subset=[label_column, 'text'], inplace=True)
        df[label_column] = df[label_column].astype(int)
    if challenge_type == 'imbalance':
        print(f"Creating imbalanced subset of size {max_train_samples}...")
        majority_class = train_df_full[label_column].mode()[0]
        n_majority_target = int(max_train_samples * severity)
        majority_df_full = train_df_full[train_df_full[label_column] == majority_class]
        minority_df_full = train_df_full[train_df_full[label_column] != majority_class]
        n_majority_actual = min(n_majority_target, len(majority_df_full))
        majority_df_sampled = majority_df_full.sample(n=n_majority_actual, random_state=42, replace=False)
        n_minority_needed = max_train_samples - n_majority_actual
        minority_df_sampled = minority_df_full.sample(n=n_minority_needed, random_state=42, replace=True)
        train_df = pd.concat([majority_df_sampled, minority_df_sampled]).sample(frac=1, random_state=42)
    else:
        train_df = train_df_full.sample(n=max_train_samples, random_state=42)
        
        if challenge_type == 'noise':
            print(f"Creating noisy label subset...")
            n_noisy = int(len(train_df) * severity * 0.4)
            noisy_indices = np.random.choice(train_df.index, size=n_noisy, replace=False)
            unique_labels = train_df[label_column].unique()
            for idx in noisy_indices:
                current_label = train_df.loc[idx, label_column]
                other_labels = [l for l in unique_labels if l != current_label]
                if other_labels: train_df.loc[idx, label_column] = np.random.choice(other_labels)
                    
        elif challenge_type == 'outliers':
            print(f"Creating outlier subset...")
            embeddings = encode_texts(train_df['text'])
            centroid = embeddings.mean(axis=0)
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            n_outliers = int(len(train_df) * severity * 0.15)
            outlier_indices = np.argsort(distances)[-n_outliers:]
            normal_indices = np.argsort(distances)[:(len(train_df) - n_outliers)]
            train_df = train_df.iloc[np.concatenate([normal_indices, outlier_indices])]
            
    print(f"Final train size: {len(train_df)}. Val size: {len(val_df)}. Test size: {len(test_df)}")
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
            X_train_embedded, _, _ = joblib.load(embeddings_cache_path)
        else:
            print("Generating and caching embeddings...")
            X_train_embedded = encode_texts(train_df[text_column])
            # For simplicity, we only cache the train embeddings in this version
            joblib.dump((X_train_embedded, None, None), embeddings_cache_path)
            
        X_train_embedded_df = pd.DataFrame(X_train_embedded, index=train_df.index)
        
        meta_extractor = MetaFeatureExtractor()
        sample_indices = np.random.choice(X_train_embedded_df.index, size=min(500, len(X_train_embedded_df)), replace=False)
        meta_features = meta_extractor.characterize(train_df, text_column, label_column, X_train_embedded_df.loc[sample_indices])
        
        meta_predictor = MetaStrategyPredictor()
        policy = meta_predictor.predict_policy(meta_features)
        
        # Check if we need to handle outliers by dropping them
        if policy.get('drop_outliers') and name == 'scienceqa_outliers':
            print("Detecting and removing outliers...")
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outlier_preds = iso_forest.fit_predict(X_train_embedded)
            
            # Save original data for comparison
            original_train_df = train_df.copy()
            original_train_data = X_train_embedded.copy()
            
            # Keep only non-outlier samples
            non_outlier_indices = outlier_preds == 1
            train_df = train_df[non_outlier_indices].reset_index(drop=True)
            X_train_embedded = X_train_embedded[non_outlier_indices]
            
            print(f"Removed {np.sum(~non_outlier_indices)} outliers. Remaining samples: {len(train_df)}")
            
            # Save processed train data without outliers
            train_df.to_csv(os.path.join(output_base_dir, name, 'processed_train.csv'), index=False)
            
            # Save embeddings of processed data
            processed_train_data_path = os.path.join(output_base_dir, name, 'processed_embeddings.joblib')
            processed_labels_path = os.path.join(output_base_dir, name, 'processed_labels.joblib')
            
            joblib.dump(X_train_embedded, processed_train_data_path)
            joblib.dump(train_df[label_column].values, processed_labels_path)
            
            # Create a processed DataFrame with embeddings for AutoGluon
            processed_df = pd.DataFrame(X_train_embedded)
            processed_df[label_column] = train_df[label_column].values
            processed_df.to_csv(os.path.join(output_base_dir, name, 'processed_train.csv'), index=False)
            
            plot_paths = analyze_and_visualize(
                name=name,
                challenge_type=config['challenge'],
                before_embeddings=original_train_data,
                before_labels=original_train_df[label_column].values,
                after_embeddings=X_train_embedded,
                after_labels=train_df[label_column].values,
                output_base_dir=output_base_dir
            )
            
            if wandb_api_key and run:
                log_data = {
                    "Distribution_Comparison": wandb.Image(plot_paths['distribution_comparison']),
                    "UMAP_Comparison": wandb.Image(plot_paths['umap_comparison'])
                }
                wandb.log(log_data)
                wandb.log({
                    "preprocessing": "outlier_removal", 
                    "samples_before": len(original_train_df),
                    "samples_after": len(train_df)
                })
            
            # Skip the rest of the preprocessing pipeline if we're just removing outliers
            print(f"--- Finished experiment for {name} with outlier removal. Check W&B run for results. ---")
            if wandb_api_key and run:
                wandb.finish()
            continue
        
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
        
        before_train_labels = train_df[label_column].values
        after_train_labels = before_train_labels.copy()

        # Special handling for ScienceQA - only use SMOTE if it's not specifically disabled for this dataset
        apply_smote = policy.get('use_smote') and not policy.get('scienceqa_special_handling', False)
        
        if apply_smote:
            print("Applying SMOTE post-transformation...")
            
            unique_classes, class_counts = np.unique(after_train_labels, return_counts=True)
            min_class_size = class_counts.min()

            if min_class_size > 1:
                k_neighbors = max(1, min_class_size - 1)
                print(f"Smallest class has {min_class_size} samples. Setting SMOTE k_neighbors to {k_neighbors}.")
                
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                processed_train_data, after_train_labels = smote.fit_resample(processed_train_data, after_train_labels)
                print(f"Data resampled with SMOTE. New train size: {len(processed_train_data)}")
            else:
                print(f"Skipping SMOTE: The smallest class has only {min_class_size} sample(s), which is not enough for SMOTE.")
        
        # Save embeddings of processed data for AutoGluon
        processed_train_data_path = os.path.join(output_base_dir, name, 'processed_embeddings.joblib')
        processed_labels_path = os.path.join(output_base_dir, name, 'processed_labels.joblib')
        
        joblib.dump(processed_train_data, processed_train_data_path)
        joblib.dump(after_train_labels, processed_labels_path)
        
        # Create a processed DataFrame with embeddings for AutoGluon
        processed_df = pd.DataFrame(processed_train_data)
        processed_df[label_column] = after_train_labels
        processed_df.to_csv(os.path.join(output_base_dir, name, 'processed_train.csv'), index=False)
        
        wandb.config.update({"dataset_name": name, "challenge_type": name.split('_')[-1], "train_samples": len(train_df), **meta_features, "policy": policy})
        wandb.log({"best_pipeline_str": str(best_pipeline_structure)})
        
        plot_paths = analyze_and_visualize(
            name=name,
            challenge_type=config['challenge'],
            before_embeddings=X_train_embedded,
            before_labels=before_train_labels,
            after_embeddings=processed_train_data,
            after_labels=after_train_labels,
            output_base_dir=output_base_dir
        )
        
        log_data = {
            "Distribution_Comparison": wandb.Image(plot_paths['distribution_comparison']),
            "UMAP_Comparison": wandb.Image(plot_paths['umap_comparison'])
        }
        if wandb_api_key and run:
            wandb.log(log_data)
        
        artifact = wandb.Artifact(name=f"preprocessor_{name}", type="preprocessing_pipeline")
        pipeline_path = os.path.join(output_base_dir, f"{name}_pipeline.joblib")
        joblib.dump(final_preprocessor, pipeline_path)
        artifact.add_file(pipeline_path)
        if wandb_api_key and run:
            run.log_artifact(artifact)
        
        print(f"--- Finished experiment for {name}. Check W&B run for results. ---")
        if wandb_api_key and run:
            wandb.finish()

main()
