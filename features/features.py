from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Config import params
import pickle

class FeatureEngineer:
    def init(self):
        self.vectorizer = TfidfVectorizer(
            max_features=params.MAX_FEATURES,
            stop_words=params.STOP_WORDS
        )

    def create_features(self, df):
        """Create TF-IDF vectors and similarity matrix"""
        tfidf_vectors = self.vectorizer.fit_transform(df['info'])
        similarity = cosine_similarity(tfidf_vectors)
        return similarity

    def save_model(self, similarity, df, paths):
        """Save model artifacts"""
        pickle.dump(similarity, open(paths.SIMILARITY_FILE, 'wb'))
        pickle.dump(df, open(paths.GAMES_LIST_FILE, 'wb'))