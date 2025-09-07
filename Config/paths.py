from pathlib import Path

class Paths:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.DATA_DIR = self.BASE_DIR / "Data" / "data"
        self.RAW_DATA = self.DATA_DIR / "GamesData.csv"
        self.REVIEWS_DATA = self.DATA_DIR / "GamesReviewData.json"
        
        # Model artifacts paths - updated structure
        self.MODEL_DIR = self.BASE_DIR / "models" / "model"
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        # Corrected file paths with clear naming
        self.SIMILARITY_MATRIX = self.MODEL_DIR / "similarity_matrix.pkl"  # 14775×14775 array
        self.GAME_METADATA = self.MODEL_DIR / "game_metadata.pkl"          # DataFrame with game info
        self.TFIDF_VECTORS = self.MODEL_DIR / "tfidf_vectors.pkl"          # Optional: TF-IDF features
        
        # Backward compatibility aliases (temporary - can remove after updating all code)
        self.SIMILARITY_FILE = self.SIMILARITY_MATRIX  # Alias
        self.GAMES_LIST_FILE = self.GAME_METADATA       # Alias

paths = Paths()