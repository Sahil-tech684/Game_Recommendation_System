# 🎮 GameFinder Pro - AI Powered Game Recommendation System

An end-to-end **Content-Based Game Recommendation System** built on 83,000+ video games scraped from RAWG, featuring an interactive Streamlit application that recommends similar games and performs AI-powered sentiment analysis on player reviews using Hugging Face Transformer models.

---

## Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Data Collection Pipeline](#-data-collection-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Recommendation Engine](#-recommendation-engine)
- [Sentiment Analysis](#-sentiment-analysis)
- [Streamlit Application](#-streamlit-application)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 📖 Overview

GameFinder Pro combines **Web Scraping**, **Natural Language Processing**, **Machine Learning**, and **Interactive Data Visualization** into one application.

The project consists of two major components:

### 🎮 Intelligent Game Recommendation Engine
Recommends games based on their textual similarity using:
- Genre
- Description
- Developers
- Publishers
- Tags
- Game Series

...instead of relying on user ratings or collaborative filtering.

### 😊 AI Review Sentiment Analysis
Analyzes player reviews using a Transformer-based sentiment model and classifies them as:
- 🟢 Positive
- ⚪ Neutral
- 🔴 Negative

...along with overall sentiment distribution.

---

## ✨ Features

### 🎯 Smart Game Recommendations
- Recommend similar games instantly
- Content-based recommendation engine
- Search from 83K+ games
- Adjustable number of recommendations

### 📄 Detailed Game Information
Each recommendation includes:
- Poster
- Description
- Genres
- Developers
- Publishers
- Similarity Score
- RAWG Page Link

### 😊 Sentiment Analysis
- Transformer-based sentiment classification
- Positive / Neutral / Negative reviews
- Confidence threshold adjustment
- Interactive sentiment visualization

### 📊 Dashboard
- Total Games
- Total Reviews
- Recommendation Statistics
- Interactive Charts

---

## 📂 Project Structure

```
GameFinder-Pro/
│
├── Config/
│   ├── params.py
│   └── paths.py
│
├── Data/
│   ├── GamesData.csv
│   ├── GamesReviewData.zip
│   └── preprocessing.py
│
├── features/
│   ├── engineering.py
│   └── features.py
│
├── models/
│   ├── training.py
│   └── inference.py
│
├── app/
│   ├── app.py
│   └── style.css
│
├── Notebook/
│   └── Games_Recommender.ipynb
│
├── RawgScraper.py
├── RawgReviewsScraper.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🕸 Data Collection Pipeline

### Step 1 — Game Metadata Scraping
Using **Scrapy**, the crawler extracts information from RAWG including:
- Game Name
- Description
- Genres
- Developers
- Publishers
- Tags
- Other Games in Series
- Poster
- Website

**Result:** `GamesData.csv` — containing approximately 83,000 games.

### Step 2 — Review Scraping
A dedicated scraper collects player reviews for every game.

The review dataset contains:
- Review Text
- Reviewer Rating
- Game ID

...which are later merged with the game metadata.

### Step 3 — Data Preprocessing
The preprocessing pipeline performs:
- Missing value handling
- Text normalization
- Lowercasing
- Stopword removal
- Porter Stemming
- Feature cleaning
- Combined metadata generation

Final features are merged into one textual column called **`info`**, which represents every game's identity.

---

## 🧠 Feature Engineering

The recommendation model converts textual metadata into numerical vectors using **TF-IDF Vectorization**.

Important parameters:
- `max_features = 5000`
- English stopwords removal
- Porter stemming

The resulting TF-IDF vectors are transformed into a **Game × Game Similarity Matrix** using:
- Linear Kernel
- Cosine Similarity

Model artifacts generated:
- `similarity_matrix.pkl`
- `game_metadata.pkl`
- `tfidf_vectors.pkl`

---

## 🤖 Recommendation Engine

The recommender follows a **Content-Based Filtering** approach.

Unlike collaborative filtering, recommendations depend entirely on the content and metadata of games.

### Available Recommendation Strategies

| Strategy | Description |
|---|---|
| Similarity Matrix | Fastest and default recommendation method |
| KNN | Memory-efficient nearest neighbor search |
| Truncated SVD + KNN | Reduced dimensionality with faster similarity search |

The inference engine supports:
- Exact search
- Case-insensitive search
- Partial matching
- Cached recommendations
- Fast O(1) game lookup

---

## 😊 Sentiment Analysis

Player reviews are analyzed using **Hugging Face Transformers**.

**Preferred model:**
`cardiffnlp/twitter-roberta-base-sentiment-latest`

**Fallback:**
`pipeline("sentiment-analysis")`

Each review is classified into:
- Positive
- Neutral
- Negative

The application displays:
- Review cards
- Sentiment labels
- Confidence scores
- Pie chart visualization

---

## 🖥 Streamlit Application

The Streamlit interface provides:

### 🎮 Game Selection
Choose any game from the dataset.

### 🎯 Recommendation Settings
Customize:
- Number of recommendations
- Similarity score visibility
- Review length
- Confidence threshold

### 📊 Recommendation Cards
Each recommendation displays:
- Poster
- Game Name
- Similarity Score
- RAWG Link

### 📈 Analytics Dashboard
Displays:
- Total Games
- Total Reviews
- Recommendation Metrics
- Sentiment Distribution

---

## 🚀 Installation

Clone the repository:
```bash
git clone https://github.com/Sahil-tech684/Game_Recommendation_System.git
cd Game_Recommendation_System
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶ Running the Project

### Option 1 — Using Pretrained Model
Place the pretrained files inside the `models` directory:
- `similarity_matrix.pkl`
- `game_metadata.pkl`
- `tfidf_vectors.pkl`

Run:
```bash
cd app
streamlit run app.py
```

### Option 2 — Train from Scratch

**Scrape Data**
```bash
python RawgScraper.py
python RawgReviewsScraper.py
```

**Build Recommendation Model**
```bash
python main.py
```

**Launch Streamlit**
```bash
cd app
streamlit run app.py
```

Open: `http://localhost:8501`

---

## 🛠 Tech Stack

| Category | Technologies |
|---|---|
| **Programming** | Python |
| **Web Scraping** | Scrapy |
| **Data Processing** | Pandas, NumPy, NLTK |
| **Machine Learning** | TF-IDF Vectorization, Cosine Similarity, Linear Kernel, Nearest Neighbors, Truncated SVD |
| **NLP** | Hugging Face Transformers, RoBERTa, PyTorch |
| **Deployment** | Streamlit |
| **Visualization** | Plotly, Custom CSS |
| **Model Persistence** | Pickle, Joblib |

---

## 📊 Project Workflow

```
RAWG Website
      │
      ▼
Scrapy Crawlers
      │
      ▼
Game Metadata + Reviews
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Similarity Matrix
      │
      ▼
Recommendation Engine
      │
      ▼
Streamlit Application
      │
      ▼
Sentiment Analysis
      │
      ▼
Final Recommendations
```

---

## 🚀 Future Improvements
- Hybrid Recommendation System
- Collaborative Filtering
- User Authentication
- Personalized Recommendations
- Cloud Deployment
- Docker Support
- REST API
- Real-time Recommendation Updates
- LLM-based Game Summaries
- Multi-language Review Analysis

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes
   ```bash
   git commit -m "Added new feature"
   ```
4. Push to GitHub
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Sahil Singh**
*Data Analyst | Python Developer | Machine Learning Enthusiast*

Passionate about Data Science, AI, Web Scraping, Recommendation Systems, and Automation.

---

⭐ If you found this project helpful, consider starring the repository!
