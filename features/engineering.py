from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Config.params import params
from Config import paths
from sklearn.metrics.pairwise import linear_kernel
import pickle

class FeatureEngineer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=params.MAX_FEATURES,
            stop_words=params.STOP_WORDS
        )
    
    def create_features(self, df):
        """Create TF-IDF vectors and similarity matrix"""
        tfidf_vectors = self.vectorizer.fit_transform(df['info'])
        similarity = linear_kernel(tfidf_vectors, tfidf_vectors)
        #similarity = cosine_similarity(tfidf_vectors)
        return similarity
    
    def save_model(self, similarity, tfidf_vectors, df, paths):
        """Save artifacts using new path structure"""
        # Save similarity matrix
        with open(paths.SIMILARITY_MATRIX, 'wb') as f:
            pickle.dump(similarity, f)
        
        # Save game metadata (as DataFrame)
        df.to_pickle(paths.GAME_METADATA)
        
        # Optionally save TF-IDF vectors
        with open(paths.TFIDF_VECTORS, 'wb') as f:
            pickle.dump(tfidf_vectors, f)