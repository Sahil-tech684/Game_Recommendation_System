import pandas as pd
import numpy as np
from Config import paths
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import List, Dict, Any
import nltk
import os
from nltk import data

# Set the NLTK data path explicitly (add before any NLTK operations)
nltk.data.path.append(os.path.expanduser("~/nltk_data"))  # Linux/Mac
nltk.data.path.append(os.path.join(os.environ['APPDATA'], 'nltk_data'))  # Windows


class DataPreprocessor:
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_string(self, text: str) -> str:
        """Clean and normalize text string"""
        if not isinstance(text, str):
            return ""
        return text.strip().replace(' ', '').lower()
    
    def stem_text(self, text: str) -> str:
        """Apply stemming to text"""
        return " ".join([self.ps.stem(word) for word in text.split()])
    
    def remove_stopwords(self, text: str) -> str:
        """Improved stopword removal with fallback"""
        try:
            # Try standard word tokenization first
            tokens = nltk.word_tokenize(text)
        except LookupError:
            # Fallback 1: Use simple whitespace tokenizer
            tokens = text.split()
        except:
            # Fallback 2: Return original text if all fails
            return text
            
        return " ".join([word for word in tokens if word.lower() not in self.stop_words])
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        # Drop unnecessary columns and handle missing values
        df = df.drop(columns=['website'], errors='ignore')
        df = df.dropna(subset=['about', 'genre'])
        
        # Clean text columns
        text_cols = ['publishers', 'genre', 'developers', 'tags', 'other_games_series']
        for col in text_cols:
            df[col] = df[col].apply(self.clean_string)
        
        # Combine features into 'info' column
        df['info'] = (
            df['about'].astype(str) + " " + 
            df['genre'].astype(str) + " " + 
            df['publishers'].astype(str) + " " + 
            df['developers'].astype(str) + " " + 
            df['tags'].astype(str) + " " + 
            df['other_games_series'].fillna('')
        )
        
        # Text processing
        df['info'] = df['info'].str.lower()
        df['info'] = df['info'].apply(self.stem_text)
        df['info'] = df['info'].apply(self.remove_stopwords)
        
        return df[['id', 'name', 'info', 'poster', 'about']]