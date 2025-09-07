from Data.preprocessing import DataPreprocessor
from features.engineering import FeatureEngineer
from Config.paths import paths
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def main():
    try:
        # 1. Load and preprocess data
        games_df = pd.read_csv(paths.RAW_DATA)
        reviews_df = pd.read_json(paths.REVIEWS_DATA)
        
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess_data(games_df)
        processed_df = pd.merge(processed_df, reviews_df, how='left', on='id')
        
        # 2. Feature engineering - UPDATED
        engineer = FeatureEngineer()
        tfidf_vectors = engineer.vectorizer.fit_transform(processed_df['info'])
        similarity_matrix = cosine_similarity(tfidf_vectors)
        
        # 3. Save artifacts - UPDATED
        engineer.save_model(
            similarity=similarity_matrix,
            tfidf_vectors=tfidf_vectors,
            df=processed_df,
            paths=paths
        )
        
        print("Pipeline executed successfully! Artifacts saved:")
        print(f"- Similarity matrix: {paths.SIMILARITY_MATRIX}")
        print(f"- Game metadata: {paths.GAME_METADATA}")
        print(f"- TF-IDF vectors: {paths.TFIDF_VECTORS}")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()