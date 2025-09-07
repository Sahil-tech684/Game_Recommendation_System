import nltk
from typing import List, Dict

def setup_nltk():
    """Download required NLTK data"""
    nltk.download('stopwords')
    nltk.download('punkt')

def extract_review_texts(reviews: List[Dict]) -> List[str]:
    """Extract text from reviews"""
    return [review['text'] for review in reviews if isinstance(reviews, list) and 'text' in review]