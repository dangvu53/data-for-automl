#!/usr/bin/env python3
"""
Meta-Learning Framework for Automated Data Processing

This framework implements a meta-learning approach to automatically discover
optimal sequences of data preprocessing, selection, and augmentation operations
across different data types (text, tabular, multimodal).

Architecture:
1. Unified Data Representation: Convert all data types to common embedding space
2. Operation Space: Modular library of preprocessing/selection/augmentation techniques  
3. Meta-Learning Search Engine: Evolutionary algorithm to discover optimal operation sequences
4. Evaluation: Use AutoGluon as the downstream ML framework for performance evaluation
"""

import os
import pandas as pd
import numpy as np
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import random
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OperationResult:
    """Result of applying an operation to data"""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    operation_name: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class OperationSequence:
    """Sequence of operations to apply to data"""
    operations: List[Dict[str, Any]]
    fitness_score: Optional[float] = None
    evaluation_metrics: Optional[Dict[str, float]] = None
    execution_time: Optional[float] = None

class DataEmbedder(ABC):
    """Abstract base class for data embedding strategies"""
    
    @abstractmethod
    def embed(self, data: pd.DataFrame) -> np.ndarray:
        """Convert data to unified embedding representation"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings"""
        pass

class TextEmbedder(DataEmbedder):
    """Text data embedder using sentence transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the sentence transformer model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded text embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence-transformers: {e}")
                # Fallback to TF-IDF
                logger.info("Falling back to TF-IDF embeddings")
                self._use_tfidf_fallback()

    def _use_tfidf_fallback(self):
        """Fallback to TF-IDF if sentence-transformers fails"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._model = TfidfVectorizer(
            max_features=384,  # Match sentence transformer dimension
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self._is_tfidf = True

    def embed(self, data: pd.DataFrame) -> np.ndarray:
        """Convert text data to embeddings"""
        self._load_model()

        # Find text column
        text_col = 'text'
        if text_col not in data.columns:
            # Try to find a text-like column
            text_cols = [col for col in data.columns if 'text' in col.lower() or 'premise' in col.lower() or 'hypothesis' in col.lower()]
            if text_cols:
                text_col = text_cols[0]
            else:
                raise ValueError("No text column found in data")

        texts = data[text_col].fillna("").astype(str).tolist()

        if hasattr(self, '_is_tfidf') and self._is_tfidf:
            # TF-IDF fallback
            if not hasattr(self._model, 'vocabulary_'):
                embeddings = self._model.fit_transform(texts)
            else:
                embeddings = self._model.transform(texts)
            return embeddings.toarray()
        else:
            # Sentence transformers
            embeddings = self._model.encode(texts, show_progress_bar=False)
            return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        if hasattr(self, '_is_tfidf') and self._is_tfidf:
            return 384  # TF-IDF fallback dimension
        else:
            self._load_model()
            return self._model.get_sentence_embedding_dimension()

class TabularEmbedder(DataEmbedder):
    """Tabular data embedder using simple feature engineering"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        
    def embed(self, data: pd.DataFrame) -> np.ndarray:
        """Convert tabular data to embeddings"""
        # Simple approach: use numerical features directly and encode categoricals
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Separate numerical and categorical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column if present
        target_cols = ['label', 'target', 'y']
        for target_col in target_cols:
            if target_col in numerical_cols:
                numerical_cols.remove(target_col)
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)
        
        features = []
        
        # Process numerical features
        if numerical_cols:
            scaler = StandardScaler()
            numerical_features = scaler.fit_transform(data[numerical_cols].fillna(0))
            features.append(numerical_features)
        
        # Process categorical features (simple label encoding)
        if categorical_cols:
            categorical_features = []
            for col in categorical_cols:
                le = LabelEncoder()
                encoded = le.fit_transform(data[col].fillna('missing').astype(str))
                categorical_features.append(encoded.reshape(-1, 1))
            
            if categorical_features:
                categorical_features = np.hstack(categorical_features)
                features.append(categorical_features)
        
        # Combine all features
        if features:
            combined_features = np.hstack(features)
        else:
            # Fallback: create dummy features
            combined_features = np.zeros((len(data), 1))
        
        # Pad or truncate to desired embedding dimension
        if combined_features.shape[1] < self.embedding_dim:
            # Pad with zeros
            padding = np.zeros((combined_features.shape[0], self.embedding_dim - combined_features.shape[1]))
            combined_features = np.hstack([combined_features, padding])
        elif combined_features.shape[1] > self.embedding_dim:
            # Truncate
            combined_features = combined_features[:, :self.embedding_dim]
        
        return combined_features
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim

class Operation(ABC):
    """Abstract base class for data operations"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
    
    @abstractmethod
    def apply(self, data: pd.DataFrame, embeddings: np.ndarray = None) -> OperationResult:
        """Apply the operation to data"""
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Get the parameter space for this operation"""
        pass

class DataSelector(Operation):
    """Base class for data selection operations"""
    pass

class DataAugmenter(Operation):
    """Base class for data augmentation operations"""
    pass

class DataPreprocessor(Operation):
    """Base class for data preprocessing operations"""
    pass

class MetaLearningFramework:
    """Main meta-learning framework for automated data processing"""
    
    def __init__(self, 
                 embedder: DataEmbedder,
                 operations: List[Operation],
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 max_sequence_length: int = 5,
                 output_dir: str = "./meta_learning_results"):
        
        self.embedder = embedder
        self.operations = operations
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_sequence_length = max_sequence_length
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize population
        self.population: List[OperationSequence] = []
        self.best_sequence: Optional[OperationSequence] = None
        self.evolution_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized MetaLearningFramework with {len(operations)} operations")
    
    def generate_random_sequence(self) -> OperationSequence:
        """Generate a random sequence of operations"""
        sequence_length = random.randint(1, self.max_sequence_length)
        operations = []
        
        for _ in range(sequence_length):
            # Select random operation
            operation = random.choice(self.operations)
            
            # Generate random parameters
            param_space = operation.get_parameter_space()
            parameters = {}
            for param_name, param_values in param_space.items():
                parameters[param_name] = random.choice(param_values)
            
            operations.append({
                'operation_class': operation.__class__.__name__,
                'parameters': parameters
            })
        
        return OperationSequence(operations=operations)
    
    def initialize_population(self):
        """Initialize the population with random operation sequences"""
        logger.info(f"Initializing population of size {self.population_size}")
        self.population = []
        
        for _ in range(self.population_size):
            sequence = self.generate_random_sequence()
            self.population.append(sequence)
        
        logger.info(f"Generated {len(self.population)} random operation sequences")
    
    def apply_sequence(self, data: pd.DataFrame, sequence: OperationSequence) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply a sequence of operations to data"""
        current_data = data.copy()
        metadata = {
            'operations_applied': [],
            'total_execution_time': 0,
            'errors': []
        }
        
        # Get initial embeddings
        try:
            embeddings = self.embedder.embed(current_data)
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            embeddings = None
        
        start_time = time.time()
        
        for op_config in sequence.operations:
            try:
                # Find operation class
                operation_class = None
                for op in self.operations:
                    if op.__class__.__name__ == op_config['operation_class']:
                        operation_class = op.__class__
                        break
                
                if operation_class is None:
                    raise ValueError(f"Operation class {op_config['operation_class']} not found")
                
                # Create operation instance with parameters
                operation = operation_class(
                    name=op_config['operation_class'],
                    parameters=op_config['parameters']
                )
                
                # Apply operation
                result = operation.apply(current_data, embeddings)
                
                if result.success:
                    current_data = result.data
                    # Update embeddings if data changed significantly
                    if len(current_data) != len(embeddings) if embeddings is not None else True:
                        try:
                            embeddings = self.embedder.embed(current_data)
                        except:
                            embeddings = None
                else:
                    metadata['errors'].append(f"{operation.name}: {result.error_message}")
                
                metadata['operations_applied'].append({
                    'operation': operation.name,
                    'parameters': operation.parameters,
                    'success': result.success,
                    'execution_time': result.execution_time
                })
                
            except Exception as e:
                logger.warning(f"Error applying operation {op_config}: {e}")
                metadata['errors'].append(f"{op_config['operation_class']}: {str(e)}")
        
        metadata['total_execution_time'] = time.time() - start_time

        return current_data, metadata

    def evaluate_sequence(self,
                         train_data: pd.DataFrame,
                         target_col: str,
                         sequence: OperationSequence,
                         quick_eval: bool = True) -> float:
        """Evaluate a sequence using ONLY training data (no data leakage)"""
        try:
            # Apply sequence to training data ONLY
            processed_train, train_metadata = self.apply_sequence(train_data, sequence)

            # Check if we have enough data after processing
            if len(processed_train) < 10:  # Need minimum samples
                return 0.0

            # Split processed training data for evaluation (no validation data used)
            from sklearn.model_selection import train_test_split
            train_subset, val_subset = train_test_split(
                processed_train, test_size=0.3, random_state=42,
                stratify=processed_train[target_col] if target_col in processed_train.columns else None
            )

            # Quick evaluation using simple models
            if quick_eval:
                return self._quick_evaluate(train_subset, val_subset, target_col)
            else:
                return self._autogluon_evaluate(train_subset, val_subset, target_col)

        except Exception as e:
            logger.warning(f"Error evaluating sequence: {e}")
            return 0.0  # Return worst possible score

    def _quick_evaluate(self, train_data: pd.DataFrame, val_data: pd.DataFrame, target_col: str) -> float:
        """Quick evaluation using simple sklearn models"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score

            # Prepare features (exclude target)
            feature_cols = [col for col in train_data.columns if col != target_col]

            if not feature_cols or len(train_data) == 0 or len(val_data) == 0:
                return 0.0

            # Make copies to avoid SettingWithCopyWarning
            X_train = train_data[feature_cols].copy()
            y_train = train_data[target_col].copy()
            X_val = val_data[feature_cols].copy()
            y_val = val_data[target_col].copy()

            # Handle categorical features simply
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    le = LabelEncoder()
                    # Fit on combined data to handle unseen categories
                    combined_values = pd.concat([X_train[col], X_val[col]]).fillna('missing').astype(str)
                    le.fit(combined_values)
                    X_train.loc[:, col] = le.transform(X_train[col].fillna('missing').astype(str))
                    X_val.loc[:, col] = le.transform(X_val[col].fillna('missing').astype(str))

            # Fill missing values
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)

            # Encode target if categorical
            if y_train.dtype == 'object':
                le_target = LabelEncoder()
                y_train = le_target.fit_transform(y_train.astype(str))
                y_val = le_target.transform(y_val.astype(str))

            # Check if we have valid data
            if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                return 0.0

            # Train simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            return accuracy

        except Exception as e:
            logger.warning(f"Quick evaluation failed: {e}")
            return 0.0

    def _autogluon_evaluate(self, train_data: pd.DataFrame, val_data: pd.DataFrame, target_col: str) -> float:
        """Full evaluation using AutoGluon (slower but more accurate)"""
        try:
            from autogluon.tabular import TabularPredictor
            import tempfile

            # Create temporary directory for AutoGluon
            with tempfile.TemporaryDirectory() as temp_dir:
                predictor = TabularPredictor(
                    label=target_col,
                    path=temp_dir,
                    verbosity=0
                )

                # Quick training with time limit
                predictor.fit(
                    train_data=train_data,
                    time_limit=60,  # 1 minute limit for quick evaluation
                    presets='medium_quality_faster_train'
                )

                # Evaluate
                performance = predictor.evaluate(val_data, silent=True)

                # Extract accuracy or main metric
                if isinstance(performance, dict):
                    return performance.get('accuracy', performance.get('acc', list(performance.values())[0]))
                else:
                    return float(performance)

        except Exception as e:
            logger.warning(f"AutoGluon evaluation failed: {e}")
            return self._quick_evaluate(train_data, val_data, target_col)

    def evolve_population(self, train_data: pd.DataFrame, target_col: str):
        """Evolve the population using genetic algorithm - ONLY uses training data"""
        evolution_start_time = time.time()
        logger.info("Starting evolution process...")

        for generation in range(self.generations):
            generation_start_time = time.time()
            logger.info(f"Generation {generation + 1}/{self.generations}")

            # Evaluate all sequences in population
            fitness_scores = []
            evaluation_start_time = time.time()
            for i, sequence in enumerate(self.population):
                if sequence.fitness_score is None:
                    fitness = self.evaluate_sequence(train_data, target_col, sequence)
                    sequence.fitness_score = fitness
                else:
                    fitness = sequence.fitness_score

                fitness_scores.append(fitness)

                if i % 5 == 0:
                    logger.info(f"Evaluated {i+1}/{len(self.population)} sequences")

            evaluation_time = time.time() - evaluation_start_time
            logger.info(f"Generation {generation + 1} evaluation time: {evaluation_time:.2f}s")

            # Track best sequence
            best_idx = np.argmax(fitness_scores)
            if self.best_sequence is None or fitness_scores[best_idx] > self.best_sequence.fitness_score:
                self.best_sequence = self.population[best_idx]
                logger.info(f"New best sequence found with fitness: {fitness_scores[best_idx]:.4f}")

            # Record generation statistics
            generation_stats = {
                'generation': generation + 1,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'worst_fitness': min(fitness_scores),
                'timestamp': datetime.now().isoformat(),
                'evaluation_time': evaluation_time
            }
            self.evolution_history.append(generation_stats)

            generation_time = time.time() - generation_start_time
            logger.info(f"Generation {generation + 1} - Best: {max(fitness_scores):.4f}, "
                       f"Avg: {np.mean(fitness_scores):.4f}, Worst: {min(fitness_scores):.4f}, "
                       f"Time: {generation_time:.2f}s")

            # Create next generation
            if generation < self.generations - 1:
                self.population = self._create_next_generation(fitness_scores)

        total_evolution_time = time.time() - evolution_start_time
        logger.info(f"Total evolution time: {total_evolution_time:.2f}s")

    def _create_next_generation(self, fitness_scores: List[float]) -> List[OperationSequence]:
        """Create next generation using selection, crossover, and mutation"""
        new_population = []

        # Keep best sequences (elitism)
        elite_count = max(1, self.population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])

        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)

            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1 if random.random() < 0.5 else parent2

            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            new_population.append(child)

        return new_population[:self.population_size]

    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> OperationSequence:
        """Select parent using tournament selection"""
        tournament_indices = random.sample(range(len(self.population)),
                                          min(tournament_size, len(self.population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]

    def _crossover(self, parent1: OperationSequence, parent2: OperationSequence) -> OperationSequence:
        """Create child through crossover of two parents"""
        # Simple crossover: take operations from both parents
        all_operations = parent1.operations + parent2.operations

        # Randomly select operations up to max length
        if len(all_operations) > self.max_sequence_length:
            selected_ops = random.sample(all_operations, self.max_sequence_length)
        else:
            selected_ops = all_operations

        return OperationSequence(operations=selected_ops)

    def _mutate(self, sequence: OperationSequence) -> OperationSequence:
        """Mutate a sequence by modifying operations or parameters"""
        new_operations = sequence.operations.copy()

        mutation_type = random.choice(['add', 'remove', 'modify', 'reorder'])

        if mutation_type == 'add' and len(new_operations) < self.max_sequence_length:
            # Add random operation
            new_op = self.generate_random_sequence().operations[0]
            new_operations.append(new_op)

        elif mutation_type == 'remove' and len(new_operations) > 1:
            # Remove random operation
            new_operations.pop(random.randint(0, len(new_operations) - 1))

        elif mutation_type == 'modify' and new_operations:
            # Modify parameters of random operation
            op_idx = random.randint(0, len(new_operations) - 1)
            operation_class_name = new_operations[op_idx]['operation_class']

            # Find operation to get parameter space
            for op in self.operations:
                if op.__class__.__name__ == operation_class_name:
                    param_space = op.get_parameter_space()
                    new_params = {}
                    for param_name, param_values in param_space.items():
                        new_params[param_name] = random.choice(param_values)
                    new_operations[op_idx]['parameters'] = new_params
                    break

        elif mutation_type == 'reorder' and len(new_operations) > 1:
            # Reorder operations
            random.shuffle(new_operations)

        return OperationSequence(operations=new_operations)

    def save_results(self):
        """Save evolution results and best sequence"""
        results = {
            'best_sequence': {
                'operations': self.best_sequence.operations if self.best_sequence else None,
                'fitness_score': self.best_sequence.fitness_score if self.best_sequence else None
            },
            'evolution_history': self.evolution_history,
            'framework_config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_sequence_length': self.max_sequence_length
            }
        }

        results_file = os.path.join(self.output_dir, 'meta_learning_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    def run_meta_learning(self,
                         train_data: pd.DataFrame,
                         target_col: str) -> OperationSequence:
        """Run the complete meta-learning process - ONLY uses training data"""
        logger.info("Starting meta-learning process...")
        logger.info(f"Training data size: {len(train_data)}")
        logger.info(f"Target column: {target_col}")

        # Initialize population
        self.initialize_population()

        # Evolve population using only training data
        self.evolve_population(train_data, target_col)

        # Save results
        self.save_results()

        if self.best_sequence:
            logger.info(f"Meta-learning completed. Best fitness: {self.best_sequence.fitness_score:.4f}")
        else:
            logger.warning("Meta-learning completed but no best sequence found")

        return self.best_sequence
