import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import pairwise_distances
import joblib
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import logging
from abc import ABC, abstractmethod
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from features.engineering import FeatureEngineer  # Update with your actual module
from Config import paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStrategy(ABC):
    """Abstract base class for different model strategies."""
    
    @abstractmethod
    def create_model(self, **kwargs):
        """Create the model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class SimilarityMatrixStrategy(ModelStrategy):
    """Strategy using precomputed similarity matrix (original approach)."""
    
    def __init__(self):
        self.model_type = "similarity_matrix"
    
    def create_model(self, similarity_matrix: np.ndarray, **kwargs):
        """Create model from similarity matrix."""
        logger.info("Using similarity matrix strategy")
        return similarity_matrix
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "type": self.model_type,
            "description": "Precomputed similarity matrix approach",
            "memory_efficient": False,
            "supports_incremental": False
        }


class NearestNeighborsStrategy(ModelStrategy):
    """Strategy using sklearn NearestNeighbors."""
    
    def __init__(self, n_neighbors: int = 10, metric: str = 'cosine'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model_type = "nearest_neighbors"
    
    def create_model(self, features: np.ndarray, **kwargs):
        """Create NearestNeighbors model."""
        logger.info(f"Creating NearestNeighbors model with {self.n_neighbors} neighbors")
        
        model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 to exclude self
            metric=self.metric,
            algorithm='auto',
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(features)
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "type": self.model_type,
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "description": "KNN-based recommendation model",
            "memory_efficient": True,
            "supports_incremental": True
        }


class PipelineStrategy(ModelStrategy):
    """Strategy using sklearn Pipeline with dimensionality reduction."""
    
    def __init__(self, n_neighbors: int = 10, n_components: int = 100, 
                 metric: str = 'cosine'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.model_type = "pipeline"
    
    def create_model(self, features: np.ndarray, **kwargs):
        """Create pipeline model with dimensionality reduction."""
        logger.info(f"Creating pipeline model with {self.n_components} components")
        
        # Ensure n_components doesn't exceed feature dimensions
        n_features = features.shape[1]
        actual_components = min(self.n_components, n_features - 1)
        
        if actual_components != self.n_components:
            logger.warning(f"Reduced components from {self.n_components} to {actual_components}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('svd', TruncatedSVD(
                n_components=actual_components,
                random_state=42
            )),
            ('nn', NearestNeighbors(
                n_neighbors=self.n_neighbors + 1,
                metric=self.metric,
                algorithm='auto',
                n_jobs=-1
            ))
        ])
        
        pipeline.fit(features)
        return pipeline
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "type": self.model_type,
            "n_neighbors": self.n_neighbors,
            "n_components": self.n_components,
            "metric": self.metric,
            "description": "Pipeline with SVD + KNN",
            "memory_efficient": True,
            "supports_incremental": False
        }


class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_similarity_matrix(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """Evaluate similarity matrix quality."""
        logger.info("Evaluating similarity matrix...")
        
        metrics = {
            'matrix_density': np.count_nonzero(similarity_matrix) / similarity_matrix.size,
            'mean_similarity': np.mean(similarity_matrix),
            'std_similarity': np.std(similarity_matrix),
            'max_similarity': np.max(similarity_matrix),
            'diagonal_check': np.allclose(np.diag(similarity_matrix), 1.0)
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def evaluate_knn_model(self, model, features: np.ndarray, 
                          sample_size: int = 100) -> Dict[str, float]:
        """Evaluate KNN model performance."""
        logger.info("Evaluating KNN model...")
        
        # Sample random points for evaluation
        n_samples = min(sample_size, features.shape[0])
        sample_indices = np.random.choice(features.shape[0], n_samples, replace=False)
        sample_features = features[sample_indices]
        
        # Measure query time
        start_time = time.time()
        distances, indices = model.kneighbors(sample_features)
        query_time = (time.time() - start_time) / n_samples
        
        metrics = {
            'avg_query_time_ms': query_time * 1000,
            'mean_distance': np.mean(distances[:, 1:]),  # Exclude self
            'std_distance': np.std(distances[:, 1:]),
            'samples_evaluated': n_samples
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all evaluation metrics."""
        return self.metrics.copy()


class ModelPersistence:
    """Handle model saving and loading with multiple formats."""
    
    @staticmethod
    def save_model(model: Any, model_path: Union[str, Path], 
                   model_type: str = "auto") -> None:
        """Save model with appropriate format."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if model_type == "auto":
            # Auto-detect based on model type
            if isinstance(model, np.ndarray):
                model_type = "numpy"
            elif hasattr(model, 'fit') and hasattr(model, 'predict'):
                model_type = "sklearn"
            else:
                model_type = "pickle"
        
        logger.info(f"Saving model with format: {model_type}")
        
        if model_type == "numpy":
            np.savez_compressed(model_path.with_suffix('.npz'), model=model)
        elif model_type == "sklearn":
            joblib.dump(model, model_path.with_suffix('.joblib'), compress=3)
        else:  # pickle
            with open(model_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_path}")
    
    @staticmethod
    def load_model(model_path: Union[str, Path], 
                   model_type: str = "auto") -> Any:
        """Load model with appropriate format."""
        model_path = Path(model_path)
        
        if model_type == "auto":
            # Auto-detect based on file extension
            if model_path.with_suffix('.npz').exists():
                model_type = "numpy"
                model_path = model_path.with_suffix('.npz')
            elif model_path.with_suffix('.joblib').exists():
                model_type = "sklearn"
                model_path = model_path.with_suffix('.joblib')
            else:
                model_type = "pickle"
                model_path = model_path.with_suffix('.pkl')
        
        logger.info(f"Loading model with format: {model_type}")
        
        if model_type == "numpy":
            data = np.load(model_path)
            return data['model']
        elif model_type == "sklearn":
            return joblib.load(model_path)
        else:  # pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)


class ModelTrainer:
    """
    Enhanced model trainer with multiple strategies and evaluation.
    
    Maintains backward compatibility while providing advanced features.
    """
    
    def __init__(self, strategy: str = "similarity", n_neighbors: int = 10,
                 n_components: Optional[int] = 100, metric: str = 'cosine',
                 evaluate: bool = True):
        """
        Initialize trainer with specified strategy.
        
        Args:
            strategy: Model strategy ('similarity', 'knn', 'pipeline')
            n_neighbors: Number of neighbors for KNN-based strategies
            n_components: Number of components for SVD (pipeline only)
            metric: Distance metric to use
            evaluate: Whether to evaluate model performance
        """
        self.strategy_name = strategy
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.evaluate_models = evaluate
        
        # Initialize strategy
        self.strategy = self._create_strategy(strategy, n_neighbors, n_components, metric)
        
        # Initialize components
        self.evaluator = ModelEvaluator() if evaluate else None
        self.persistence = ModelPersistence()
        
        # Training state
        self.model = None
        self.training_metrics = {}
        
        logger.info(f"ModelTrainer initialized with {strategy} strategy")
    
    def _create_strategy(self, strategy: str, n_neighbors: int, 
                        n_components: Optional[int], metric: str) -> ModelStrategy:
        """Create the specified strategy."""
        if strategy == "similarity":
            return SimilarityMatrixStrategy()
        elif strategy == "knn":
            return NearestNeighborsStrategy(n_neighbors, metric)
        elif strategy == "pipeline":
            return PipelineStrategy(n_neighbors, n_components or 100, metric)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def train_model(self, features: Union[np.ndarray, Any], 
                   similarity_matrix: Optional[np.ndarray] = None) -> Any:
        """
        Train model using the configured strategy.
        
        Args:
            features: Feature matrix or TF-IDF vectors
            similarity_matrix: Precomputed similarity matrix (for similarity strategy)
            
        Returns:
            Trained model
        """
        logger.info(f"Training model with {self.strategy_name} strategy...")
        
        start_time = time.time()
        
        # Train based on strategy
        if self.strategy_name == "similarity":
            if similarity_matrix is None:
                raise ValueError("Similarity matrix required for similarity strategy")
            self.model = self.strategy.create_model(similarity_matrix)
        else:
            self.model = self.strategy.create_model(features)
        
        training_time = time.time() - start_time
        
        # Store training metrics
        self.training_metrics = {
            'training_time_seconds': training_time,
            'strategy': self.strategy_name,
            'model_info': self.strategy.get_model_info()
        }
        
        # Evaluate model if requested
        if self.evaluate_models and self.evaluator:
            if self.strategy_name == "similarity":
                eval_metrics = self.evaluator.evaluate_similarity_matrix(similarity_matrix)
            else:
                eval_metrics = self.evaluator.evaluate_knn_model(self.model, features)
            
            self.training_metrics.update(eval_metrics)
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        return self.model
    
    def save_model(self, model_path: Union[str, Path], 
                   include_metadata: bool = True) -> None:
        """
        Save trained model with metadata.
        
        Args:
            model_path: Path to save the model
            include_metadata: Whether to save training metadata
        """
        if self.model is None:
            raise ValueError("No trained model to save. Call train_model first.")
        
        model_path = Path(model_path)
        
        # Save the model
        self.persistence.save_model(self.model, model_path)
        
        # Save metadata if requested
        if include_metadata:
            metadata_path = model_path.with_suffix('.metadata.json')
            import json
            
            metadata = {
                'strategy': self.strategy_name,
                'training_metrics': self.training_metrics,
                'model_info': self.strategy.get_model_info()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: Union[str, Path]) -> Any:
        """Load a previously trained model."""
        self.model = self.persistence.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.model
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics and evaluation results."""
        return self.training_metrics.copy()
    
    def switch_strategy(self, strategy: str, **kwargs) -> None:
        """Switch to a different training strategy."""
        self.strategy_name = strategy
        self.strategy = self._create_strategy(
            strategy, 
            kwargs.get('n_neighbors', self.n_neighbors),
            kwargs.get('n_components', 100),
            kwargs.get('metric', self.metric)
        )
        logger.info(f"Switched to {strategy} strategy")


# Factory functions for easy instantiation
def create_similarity_trainer(evaluate: bool = True) -> ModelTrainer:
    """Create trainer for similarity matrix strategy."""
    return ModelTrainer(strategy="similarity", evaluate=evaluate)


def create_knn_trainer(n_neighbors: int = 10, metric: str = 'cosine',
                      evaluate: bool = True) -> ModelTrainer:
    """Create trainer for KNN strategy."""
    return ModelTrainer(strategy="knn", n_neighbors=n_neighbors, 
                       metric=metric, evaluate=evaluate)


def create_pipeline_trainer(n_neighbors: int = 10, n_components: int = 100,
                           metric: str = 'cosine', evaluate: bool = True) -> ModelTrainer:
    """Create trainer for pipeline strategy."""
    return ModelTrainer(strategy="pipeline", n_neighbors=n_neighbors,
                       n_components=n_components, metric=metric, evaluate=evaluate)


# Example usage and testing
if __name__ == "__main__":
    # 1. Load your preprocessed data
    df = pd.read_pickle("path/to/your/preprocessed_data.pkl")  # Update path
    # 2. Initialize feature engineer
    engineer = FeatureEngineer()
    # 3. Create features - THIS IS THE CRUCIAL CHANGE
    print("Creating TF-IDF features...")
    tfidf_vectors = engineer.vectorizer.fit_transform(df['info'])
    print("Computing cosine similarity...")
    similarity_matrix = cosine_similarity(tfidf_vectors)
    # 4. Train model using the similarity matrix
    print("Training model...")
    trainer = create_similarity_trainer()
    model = trainer.train_model(features=tfidf_vectors, similarity_matrix=similarity_matrix)
    # 5. Save all components properly
    print("Saving artifacts...")
    # Save the trained model
    trainer.save_model("models/model/recommender_model.pkl")
    # Save additional artifacts using FeatureEngineer
    engineer.save_model(
        similarity=similarity_matrix,
        tfidf_vectors=tfidf_vectors,
        df=df,
        paths=paths  # Your paths configuration
    )
    print("Training complete!")