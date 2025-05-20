import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
from io import StringIO


# Environment Setup

from dotenv import load_dotenv
load_dotenv()


# Background Styling

st.markdown("""
<style>
.stApp{
    background-image: url("https://images.unsplash.com/photo-1458560871784-56d23406c091?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)  

  
# HuggingFace Setup

import os
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)


## Loading the lyrics through a GitHub


def load_lyrics():
    url = "https://raw.githubusercontent.com/walkerkq/musiclyrics/master/billboard_lyrics_1964-2015.csv"
    
    try:
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
        
        df = df.rename(columns={"Lyrics": "lyrics", "Artist": "artist", "Song": "title"})
        df = df[['title','artist','lyrics']].dropna()
        
        df['mood'] = np.random.choice(['positive', 'neutral', 'negative'], size=len(df))
        
        return df.head(100)  
        
    except Exception as e:
        st.error(f"Using fallback dataset because: {str(e)}")
        return pd.DataFrame([
            {"title": "Happy", "artist": "Pharrell Williams", "lyrics": "Clap along...", "mood": "positive", "genre": "pop"},
            {"title": "Yesterday", "artist": "The Beatles", "lyrics": "All my troubles...", "mood": "negative", "genre": "rock"},
            {"title": "Blinding Lights", "artist": "The Weeknd", "lyrics": "I've been tryna...", "mood": "neutral", "genre": "electronic"}
        ])

lyrics_db = load_lyrics()




# Embedding Model 

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()
lyrics_embeddings = model.encode(lyrics_db['lyrics'].tolist())
 



# Recommendation Function 

def recommend_songs(user_input, n=3):
    sentiment = client.text_classification(text=user_input)
    mood = sentiment[0]['label'].lower()
    
    mood_matches = lyrics_db[lyrics_db["mood"] == mood]
    if len(mood_matches) == 0:
        return lyrics_db.sample(min(n, len(lyrics_db)))
    
    input_embedding = model.encode([user_input])
    mood_matches = mood_matches.reset_index(drop=True)
    similarities = cosine_similarity(input_embedding, lyrics_embeddings[mood_matches.index])[0]

    top_indices = np.argsort(similarities)[-n:][::-1]
    return mood_matches.iloc[top_indices]





# --- Streamlit UI ---
st.title("Mood-Based Song Recommender")
user_input = st.text_input("How are you feeling today?", "I feel joyful")


if st.button("Get Recommendations"):
    recommendations = recommend_songs(user_input, n=3)
    

    st.subheader("Top Recommendations:")
    cols = st.columns(3)
    for i, (_, row) in enumerate(recommendations.iterrows()):

        with cols[i]:
            st.write(f"{row['title']} by {row['artist']}")
            yt_link = f"https://www.youtube.com/results?search_query={row['title']}+{row['artist']}"
            st.markdown(f'<a href="{yt_link}" target="_blank">▶ Play on YouTube</a>',unsafe_allow_html=True)


