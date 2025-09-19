# Mood-Based Song Recommender

Advanced Streamlit app for mood-based, emotion-driven music playlist recommendations.  
Features emotion detection (BERT), genre/language filtering, Spotify audio analysis, user feedback, and data visualizations.

## Run Instructions

1. Clone repo and install requirements: `pip install -r requirements.txt`
2. Create `.env` with Spotify credentials.
3. Download lyrics dataset to `data/lyrics.csv`
4. Run: `streamlit run app.py`

## Technologies

- Streamlit
- Transformers, SentenceTransformer
- Spotify API
- Matplotlib, Scikit-learn
