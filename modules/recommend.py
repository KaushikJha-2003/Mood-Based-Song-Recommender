from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from modules.spotify_features import add_audio_features
import numpy as np
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_playlist(user_input, lyrics_db, n, artist, year, activity, emotion):
    db = lyrics_db.copy()
    if artist != "All":
        db = db[db['Artist'] == artist]
    if year != "All":
        db = db[db['Year'] == year]
    
    # Check if any songs remain after filtering
    if db.empty:
        return pd.DataFrame()  # Return empty DataFrame
    
    embs = model.encode(db['Lyrics'].tolist())
    input_emb = model.encode([user_input])
    sims = cosine_similarity(input_emb, embs)[0]
    db = db.assign(similarity=sims)
    topn = db.sort_values("similarity", ascending=False).head(n)
    topn = add_audio_features(topn)
    topn = topn.assign(emotion=emotion)
    return topn
