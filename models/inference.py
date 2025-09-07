import pickle
import numpy as np
from typing import List, Dict, Optional
import logging
from Config import paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameRecommender:
    """Optimized game recommender with enhanced performance and error handling."""
    
    def __init__(self, cache_size: int = 100):
        """
        Initialize the recommender.
        
        Args:
            cache_size: Maximum number of cached recommendations
        """
        self.similarity = None
        self.games_df = None
        self._name_to_index = {}
        self._recommendation_cache = {}
        self._cache_size = cache_size
        self._model_loaded = False
        
    def load_model(self):
        """Load trained model artifacts with error handling."""
        try:
            logger.info("Loading game recommendation model...")
            
            with open(paths.SIMILARITY_FILE, 'rb') as f:
                self.similarity = pickle.load(f)
            
            with open(paths.GAMES_LIST_FILE, 'rb') as f:
                self.games_df = pickle.load(f)
            
            # Build optimized name index for faster lookups
            self._build_name_index()
            self._model_loaded = True
            
            logger.info(f"Model loaded successfully: {len(self.games_df)} games indexed")
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _build_name_index(self):
        """Build optimized name-to-index mapping for O(1) lookups."""
        self._name_to_index = {}
        for idx, name in enumerate(self.games_df['name']):
            # Store exact match
            self._name_to_index[name] = idx
            # Store lowercase for case-insensitive search
            self._name_to_index[name.lower()] = idx
    
    def _find_game_index(self, game_name: str) -> Optional[int]:
        """
        Find game index with fallback strategies.
        
        Args:
            game_name: Name of the game to find
            
        Returns:
            Game index or None if not found
        """
        # Direct lookup (fastest)
        if game_name in self._name_to_index:
            return self._name_to_index[game_name]
        
        # Case-insensitive lookup
        game_lower = game_name.lower()
        if game_lower in self._name_to_index:
            return self._name_to_index[game_lower]
        
        # Partial match fallback (slower but more flexible)
        return self._find_partial_match(game_name)
    
    def _find_partial_match(self, game_name: str) -> Optional[int]:
        """Find partial match for game name."""
        game_lower = game_name.lower()
        
        # Check if query is contained in any game name
        for name, idx in self._name_to_index.items():
            if isinstance(name, str) and game_lower in name.lower():
                return idx
        
        # Check if any game name is contained in query
        for name, idx in self._name_to_index.items():
            if isinstance(name, str) and name.lower() in game_lower:
                return idx
        
        return None
    
    def _get_cached_recommendations(self, cache_key: str) -> Optional[List[Dict]]:
        """Get recommendations from cache."""
        return self._recommendation_cache.get(cache_key)
    
    def _cache_recommendations(self, cache_key: str, recommendations: List[Dict]):
        """Cache recommendations with LRU eviction."""
        if len(self._recommendation_cache) >= self._cache_size:
            # Simple FIFO eviction (can be enhanced to LRU if needed)
            oldest_key = next(iter(self._recommendation_cache))
            del self._recommendation_cache[oldest_key]
        
        self._recommendation_cache[cache_key] = recommendations
    
    def _format_about_field(self, about_field) -> str:
        """Format the about field consistently."""
        if isinstance(about_field, list) and len(about_field) > 0:
            return str(about_field[0])
        elif about_field is not None:
            return str(about_field)
        else:
            return "No description available"
    
    def recommend(self, game_name: str, n_recommendations: int = 10, 
                 use_cache: bool = True) -> List[Dict]:
        """
        Generate game recommendations with optimizations.
        
        Args:
            game_name: Name of the game to get recommendations for
            n_recommendations: Number of recommendations to return
            use_cache: Whether to use cached results
            
        Returns:
            List of recommended games with details
        """
        # Lazy loading
        if not self._model_loaded:
            self.load_model()
        
        # Check cache first
        cache_key = f"{game_name}_{n_recommendations}"
        if use_cache:
            cached_result = self._get_cached_recommendations(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for '{game_name}'")
                return cached_result
        
        # Find game index using optimized lookup
        game_index = self._find_game_index(game_name)
        if game_index is None:
            logger.warning(f"Game not found: '{game_name}'")
            return []
        
        try:
            # Get similarity scores for the game
            similarities = self.similarity[game_index]
            
            # Use numpy for faster sorting
            indices = np.argsort(similarities)[::-1]
            
            # Build recommendations efficiently
            recommendations = []
            count = 0
            
            for idx in indices:
                if idx != game_index and count < n_recommendations:
                    game = self.games_df.iloc[idx]
                    recommendations.append({
                        'name': game['name'],
                        'poster': game.get('poster', ''),
                        'about': self._format_about_field(game.get('about', ''))
                    })
                    count += 1
            
            # Cache the result
            if use_cache:
                self._cache_recommendations(cache_key, recommendations)
            
            logger.debug(f"Generated {len(recommendations)} recommendations for '{game_name}'")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for '{game_name}': {e}")
            return []
    
    def search_games(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for games by name.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching games
        """
        if not self._model_loaded:
            self.load_model()
        
        query_lower = query.lower()
        matches = []
        
        for idx, name in enumerate(self.games_df['name']):
            if query_lower in name.lower():
                game = self.games_df.iloc[idx]
                matches.append({
                    'name': game['name'],
                    'poster': game.get('poster', ''),
                    'about': self._format_about_field(game.get('about', ''))
                })
                
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_game_info(self, game_name: str) -> Optional[Dict]:
        """
        Get information about a specific game.
        
        Args:
            game_name: Name of the game
            
        Returns:
            Game information or None if not found
        """
        if not self._model_loaded:
            self.load_model()
        
        game_index = self._find_game_index(game_name)
        if game_index is None:
            return None
        
        game = self.games_df.iloc[game_index]
        return {
            'name': game['name'],
            'poster': game.get('poster', ''),
            'about': self._format_about_field(game.get('about', ''))
        }
    
    def clear_cache(self):
        """Clear the recommendation cache."""
        self._recommendation_cache.clear()
        logger.info("Recommendation cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cache_size': len(self._recommendation_cache),
            'max_cache_size': self._cache_size,
            'cached_queries': list(self._recommendation_cache.keys())
        }
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded
    
    def __len__(self) -> int:
        """Return number of games in the dataset."""
        return len(self.games_df) if self._model_loaded else 0
    
    def __contains__(self, game_name: str) -> bool:
        """Check if a game exists in the dataset."""
        if not self._model_loaded:
            self.load_model()
        return self._find_game_index(game_name) is not None