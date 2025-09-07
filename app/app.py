import pickle
import pandas as pd
import streamlit as st
import numpy as np
from transformers import pipeline
from typing import List, Tuple, Optional
import logging as log
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time


# Configure page
st.set_page_config(
    page_title="GameFinder Pro",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/gamefinder',
        'Report a bug': "https://github.com/yourusername/gamefinder/issues",
        'About': "# GameFinder Pro\nDiscover your next favorite game with AI-powered recommendations!"
    }
)

# Initialize sentiment analysis with error handling
@st.cache_resource
def load_sentiment_model():
    try:
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    except Exception as e:
        st.warning(f"Advanced sentiment model not available, using default: {e}")
        return pipeline("sentiment-analysis")

# Load data with comprehensive error handling
@st.cache_data
def load_data() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    try:
        # Check if files exist
        metadata_path = Path('models/model/game_metadata.pkl')
        similarity_path = Path('models/model/similarity_matrix.pkl')
        
        if not metadata_path.exists():
            st.error(f"❌ Game metadata file not found: {metadata_path}")
            st.info("Please ensure the model files are in the correct directory structure.")
            return None, None
            
        if not similarity_path.exists():
            st.error(f"❌ Similarity matrix file not found: {similarity_path}")
            return None, None
        
        # Load game metadata
        with st.spinner("🔄 Loading game database..."):
            df = pd.read_pickle(metadata_path)
        
        # Handle duplicate columns
        if 'name_x' in df.columns and 'name_y' in df.columns:
            df['name'] = df['name_x'].fillna(df['name_y'])
            df = df.drop(columns=['name_x', 'name_y'], errors='ignore')
        elif 'name_x' in df.columns:
            df['name'] = df['name_x']
            df = df.drop(columns=['name_x'], errors='ignore')
        elif 'name_y' in df.columns:
            df['name'] = df['name_y']
            df = df.drop(columns=['name_y'], errors='ignore')
        
        # Ensure required columns exist
        required_columns = ['name', 'id', 'poster']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return None, None
        
        # Clean data
        df = df.dropna(subset=['name'])
        df['name'] = df['name'].astype(str)
        df = df[df['name'].str.strip() != '']
        
        # Load similarity matrix
        with st.spinner("🔄 Loading AI recommendation engine..."):
            with open(similarity_path, 'rb') as f:
                similarity = pickle.load(f)
        
        # Validate data consistency
        if len(df) != similarity.shape[0]:
            st.error(f"❌ Data mismatch: {len(df)} games in metadata but {similarity.shape[0]} in similarity matrix")
            return None, None
        
        st.success(f"✅ Successfully loaded {len(df)} games with AI recommendations!")
        return df, similarity
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please check your model files and try again.")
        return None, None

# Enhanced recommendation function
def recommend(game: str, df: pd.DataFrame, similarity: np.ndarray, num_recommendations: int = 10) -> Tuple[List[str], List[str], List[str], List[float]]:
    try:
        # Find game index
        game_indices = df[df['name'].str.lower() == game.lower()].index
        if len(game_indices) == 0:
            return [], [], [], []
            
        index = int(game_indices[0])
        
        if index >= similarity.shape[0]:
            return [], [], [], []
        
        # Get similarity scores
        if hasattr(similarity, 'toarray'):
            distances = similarity[index].toarray().flatten()
        else:
            distances = similarity[index]
        
        # Get top recommendations (excluding the game itself)
        recommended_indices = np.argsort(-distances)[1:num_recommendations+1]
        valid_indices = [int(i) for i in recommended_indices if i < len(df)]
        
        # Extract recommendation data
        recommended_games = df.iloc[valid_indices]
        names = recommended_games['name'].tolist()
        urls = [f"https://rawg.io/games/{game_id}" for game_id in recommended_games['id'].tolist()]
        posters = recommended_games['poster'].fillna('https://via.placeholder.com/300x180?text=No+Image').tolist()
        scores = [float(distances[i]) for i in valid_indices]
        
        return names, urls, posters, scores
        
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")
        return [], [], [], []

# Enhanced CSS styling
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        .stApp {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        /* Main container */
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        /* Header styling */
        .app-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .app-subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 0 20px 20px 0;
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Cards */
        .custom-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            border: 1px solid rgba(102, 126, 234, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .custom-card:before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        .custom-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
        }
        
        /* Game cards grid */
        .game-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .game-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            border: 1px solid rgba(102, 126, 234, 0.1);
            position: relative;
        }
        
        .game-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        }
        
        .game-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            transition: transform 0.3s ease;
        }
        
        .game-card:hover img {
            transform: scale(1.05);
        }
        
        .game-card-content {
            padding: 1.2rem;
        }
        
        .game-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
            line-height: 1.3;
        }
        
        .similarity-score {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
            display: inline-block;
            margin-bottom: 0.8rem;
        }
        
        /* Review cards */
        .review-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            border-left: 4px solid;
            transition: all 0.3s ease;
        }
        
        .review-card:hover {
            transform: translateX(5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        }
        
        .positive-review {
            border-left-color: #10B981;
            background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
        }
        
        .negative-review {
            border-left-color: #EF4444;
            background: linear-gradient(135deg, #FEF2F2 0%, #FFF5F5 100%);
        }
        
        .neutral-review {
            border-left-color: #F59E0B;
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
            filter: brightness(110%);
        }
        
        /* Metrics */
        .metric-container {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(102, 126, 234, 0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Loading spinner */
        .stSpinner > div {
            border: 4px solid rgba(102, 126, 234, 0.2);
            border-radius: 50%;
            border-top: 4px solid #667eea;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        /* Progress bars */
        .stProgress .st-bo {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
            border: 1px solid #10B981;
            border-radius: 10px;
        }
        
        .stError {
            background: linear-gradient(135deg, #FEF2F2 0%, #FFF5F5 100%);
            border: 1px solid #EF4444;
            border-radius: 10px;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .game-grid {
                grid-template-columns: 1fr;
            }
            .app-header {
                font-size: 2rem;
            }
            .main-container {
                margin: 0.5rem;
                padding: 1rem;
            }
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        }
    </style>
    """, unsafe_allow_html=True)

def create_game_card(name: str, poster: str, url: str, score: float = None) -> str:
    score_html = f'<div class="similarity-score">Match: {score:.1%}</div>' if score else ''
    return f"""
    <div class="game-card fade-in">
        <a href="{url}" target="_blank" rel="noopener noreferrer">
            <img src="{poster}" alt="{name}" 
                 onerror="this.src='https://via.placeholder.com/300x200/667eea/white?text=Game+Image'">
        </a>
        <div class="game-card-content">
            {score_html}
            <h3 class="game-title">{name}</h3>
            <a href="{url}" target="_blank" rel="noopener noreferrer" 
               style="color: #667eea; text-decoration: none; font-weight: 500;">
                🎮 View Details →
            </a>
        </div>
    </div>
    """

def create_review_card(review: str, sentiment: str, score: float) -> str:
    sentiment_mapping = {
        "POSITIVE": ("😊", "#10B981", "positive-review"),
        "NEGATIVE": ("😞", "#EF4444", "negative-review"),
        "NEUTRAL": ("😐", "#F59E0B", "neutral-review")
    }
    
    emoji, color, css_class = sentiment_mapping.get(sentiment, ("😐", "#F59E0B", "neutral-review"))
    
    return f"""
    <div class="review-card {css_class}">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 1.5rem; margin-right: 10px;">{emoji}</span>
            <span style="font-weight: 600; color: {color};">{sentiment}</span>
            <span style="margin-left: 10px; background: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem;">
                {score:.1%}
            </span>
        </div>
        <p style="margin: 0; color: #444; line-height: 1.6;">{review}</p>
    </div>
    """

def create_stats_dashboard(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Games Available</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate total reviews safely
        total_reviews = 0
        if 'reviews' in df.columns:
            for reviews in df['reviews']:
                if isinstance(reviews, list):
                    total_reviews += len(reviews)
                elif pd.notna(reviews):
                    total_reviews += 1
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_reviews:,}</div>
            <div class="metric-label">Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Inject custom CSS
    inject_custom_css()
    
    # Main header
    st.markdown("""
        <div class="main-container">
            <h1 style="text-align: center; color: #667eea; font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;">🎮 GameFinder Pro</h1>
            <p class="app-subtitle">Discover your next favorite game with AI-powered recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Number of recommendations
        num_recommendations = st.slider(
            "🎯 Number of Recommendations", 
            min_value=5, 
            max_value=20, 
            value=10
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            show_similarity_scores = st.checkbox("Show Similarity Scores", value=True)
            max_review_length = st.slider("Max Review Length", 100, 1000, 512)
            sentiment_threshold = st.slider("Sentiment Confidence", 0.5, 0.9, 0.7)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI-powered recommendation system analyzes thousands of games to find perfect matches for your taste.
        
        **Features:**
        - 🤖 Machine Learning Recommendations
        - 📊 Sentiment Analysis
        - 🎨 Beautiful UI
        - 📱 Responsive Design
        """)
    
    # Load data
    with st.spinner("🚀 Initializing GameFinder Pro..."):
        df, similarity = load_data()
    
    if df is None or similarity is None:
        st.error("❌ Could not load the game database. Please check your model files.")
        return
    
    # Initialize sentiment analysis
    try:
        sentiment_analysis = load_sentiment_model()
    except Exception as e:
        st.error(f"❌ Could not load sentiment analysis model: {e}")
        return
    
    # Stats dashboard
    st.markdown("### 📊 Database Overview")
    create_stats_dashboard(df)
    
    st.markdown("---")
    
    filtered_games = df['name'].values
    
    # Game selection
    selected_game = st.selectbox(
        "🎮 Select your favorite game:",
        filtered_games,
        help="Choose a game you love to get personalized recommendations",
        key="game_select"
    )
    
    # Recommendation button with enhanced styling
    if st.button('✨ Get AI Recommendations', use_container_width=True, type="primary"):
        if not selected_game:
            st.warning("⚠️ Please select a game first!")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Analyze selected game
        status_text.text("🔍 Analyzing your selected game...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        # Step 2: Generate recommendations
        status_text.text("🤖 Generating AI recommendations...")
        progress_bar.progress(60)
        
        recommended_names, recommended_urls, recommended_posters, similarity_scores = recommend(
            selected_game, df, similarity, num_recommendations
        )
        
        if not recommended_names:
            st.error("❌ Could not generate recommendations for this game. Please try another.")
            return
        
        # Step 3: Prepare display
        status_text.text("🎨 Preparing beautiful display...")
        progress_bar.progress(80)
        time.sleep(0.3)
        
        game_details = df[df['name'].str.lower() == selected_game.lower()].iloc[0]
        
        progress_bar.progress(100)
        status_text.text("✅ Ready!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Selected game section
        st.markdown("---")
        st.markdown(f"### 🎯 Selected Game: {game_details['name']}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center;">
                <a href="https://rawg.io/games/{game_details['id']}" target="_blank" rel="noopener noreferrer">
                    <img src="{game_details['poster']}" 
                         style="border-radius: 15px; width: 100%; max-width: 350px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);"
                         onerror="this.src='https://via.placeholder.com/350x500/667eea/white?text=Game+Cover'">
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            about_text = game_details.get('about', 'No description available.')
            if isinstance(about_text, list):
                about_text = ' '.join(about_text)
            
            st.markdown(f"""
            <div class="custom-card">
                <h3 style="margin-top: 0; color: #667eea;">📖 About This Game</h3>
                <p style="line-height: 1.6; color: #444;">{about_text}</p>
                <div style="margin-top: 1.5rem;">
                    <a href="https://rawg.io/games/{game_details['id']}" target="_blank" 
                       rel="noopener noreferrer"
                       style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              color: white; text-decoration: none; padding: 0.7rem 1.5rem; 
                              border-radius: 25px; font-weight: 600; display: inline-block;
                              transition: all 0.3s ease;">
                        🔗 View on RAWG
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations section
        st.markdown("---")
        st.markdown("### 🎮 AI-Powered Recommendations")
        st.markdown("Based on advanced machine learning analysis, here are games you'll love:")
        
        # Create game cards with similarity scores
        game_cards = ""
        for name, poster, url, score in zip(recommended_names, recommended_posters, recommended_urls, similarity_scores):
            if show_similarity_scores:
                game_cards += create_game_card(name, poster, url, score)
            else:
                game_cards += create_game_card(name, poster, url)
        
        st.markdown(f"""
        <div class="game-grid">
            {game_cards}
        </div>
        """, unsafe_allow_html=True)
        
        # Reviews section with sentiment analysis
        st.markdown("---")
        st.markdown("### 📝 Player Reviews Analysis")
        
        reviews = game_details.get('reviews', [])
        
        # Handle different data types for reviews
        if reviews is None or (hasattr(reviews, '__len__') and len(reviews) == 0):
            reviews = []
        elif isinstance(reviews, (list, np.ndarray)):
            # It's already a list/array, keep it as is
            pass
        elif pd.isna(reviews):
            reviews = []
        elif isinstance(reviews, str):
            reviews = [reviews] if reviews.strip() else []
        else:
            # Convert other types to list
            try:
                reviews = list(reviews)
            except:
                reviews = []

        if reviews and len(reviews) > 0:
            with st.expander(f"📊 Sentiment Analysis ({len(reviews)} reviews)", expanded=False):
                # Debug information
                st.write(f"🔍 **Debug Info:** Found {len(reviews)} reviews, analyzing first 20...")
                
                # Analyze all reviews
                sentiments = []
                review_cards_html = ""
                processed_count = 0
                valid_reviews = 0
                
                for i, review in enumerate(reviews[:20]):  # Limit to first 20 reviews for performance
                    processed_count += 1
                    
                    # Convert review to string if needed
                    if not isinstance(review, str):
                        try:
                            review = str(review)
                        except:
                            continue
                    
                    # Clean and validate review
                    review = review.strip()
                    if len(review) == 0 or len(review) < 10:  # Skip very short reviews
                        continue
                    
                    valid_reviews += 1
                    
                    # Truncate review if too long
                    truncated_review = (review[:max_review_length] + '...') if len(review) > max_review_length else review
                    
                    try:
                        sentiment_result = sentiment_analysis(truncated_review)
                        sentiment_label = sentiment_result[0]['label']
                        sentiment_score = sentiment_result[0]['score']
                        
                        # Debug: Show first few sentiment results
                        if i < 3:
                            st.write(f"Review {i+1}: {sentiment_label} ({sentiment_score:.2f}) - '{truncated_review[:100]}...'")
                        
                        # Map different label formats to standard format
                        if sentiment_label in ['LABEL_0', 'NEGATIVE']:
                            sentiment_label = 'NEGATIVE'
                        elif sentiment_label in ['LABEL_1', 'NEUTRAL']:
                            sentiment_label = 'NEUTRAL' 
                        elif sentiment_label in ['LABEL_2', 'POSITIVE']:
                            sentiment_label = 'POSITIVE'
                        
                        # Lower the confidence threshold to show more reviews
                        confidence_threshold = max(0.3, sentiment_threshold - 0.2)
                        
                        if sentiment_score >= confidence_threshold:
                            sentiments.append(sentiment_label)
                            review_cards_html += create_review_card(review, sentiment_label, sentiment_score)
                    
                    except Exception as e:
                        st.write(f"⚠️ Error processing review {i+1}: {str(e)}")
                        continue
                
                # Show processing stats
                st.write(f"📊 **Processing Stats:** {processed_count} processed, {valid_reviews} valid, {len(sentiments)} analyzed")
                
                # Sentiment distribution
                if sentiments:
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Create a more robust pie chart
                        try:
                            fig = px.pie(
                                values=sentiment_counts.values, 
                                names=sentiment_counts.index,
                                title="Sentiment Distribution",
                                color_discrete_map={
                                    'POSITIVE': '#10B981',
                                    'NEGATIVE': '#EF4444', 
                                    'NEUTRAL': '#F59E0B'
                                }
                            )
                            fig.update_layout(
                                height=300, 
                                showlegend=True,
                                font=dict(size=12),
                                title_font_size=14
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating chart: {e}")
                            # Fallback to simple metrics
                            for sentiment, count in sentiment_counts.items():
                                st.metric(sentiment, count)
                    
                    with col2:
                        st.markdown("#### 📋 Review Highlights")
                        if review_cards_html:
                            st.markdown(f"""
                            <div style="max-height: 400px; overflow-y: auto; padding: 1rem; border: 1px solid #ddd; border-radius: 10px;">
                                {review_cards_html}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No review cards generated.")
                else:
                    st.warning(f"⚠️ No reviews met the confidence threshold of {sentiment_threshold:.1f}. Try lowering the threshold in the sidebar.")
                    
                    # Show a few raw reviews for debugging
                    if valid_reviews > 0:
                        st.write("**Sample reviews found:**")
                        for i, review in enumerate(reviews[:3]):
                            if isinstance(review, str) and len(review.strip()) > 10:
                                st.write(f"{i+1}. {str(review)[:200]}...")
        else:
            st.info("📝 No reviews available for this game.")
            
            # Debug: Show what we actually got
            original_reviews = game_details.get('reviews')
            st.write(f"🔍 **Debug:** Original reviews data type: {type(original_reviews)}")
            if hasattr(original_reviews, '__len__'):
                try:
                    st.write(f"🔍 **Debug:** Length: {len(original_reviews)}")
                except:
                    pass
            if original_reviews is not None:
                st.write(f"🔍 **Debug:** Sample content: {str(original_reviews)[:200]}...")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
            Made with ❤️ using Streamlit • Powered by AI and Machine Learning
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()