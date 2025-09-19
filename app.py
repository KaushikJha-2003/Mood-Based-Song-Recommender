import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from modules.emotion import get_emotion
from modules.recommend import recommend_playlist
from modules.visualizations import show_mood_chart, show_tsne
from dotenv import load_dotenv
import os

# --- ENV SETUP ---
load_dotenv()
# --- Data ---
@st.cache_data
def load_lyrics():
    df = pd.read_csv("data/lyrics.csv")
    return df
lyrics_db = load_lyrics()

# --- Session ---
if "history" not in st.session_state: st.session_state.history = []
if "ratings" not in st.session_state: st.session_state.ratings = {}
if "last_playlist" not in st.session_state: st.session_state.last_playlist = pd.DataFrame()

# --- UI ---
st.title("🎵 Mood-Based Music Recommender (Advanced)")
artist = st.sidebar.selectbox("Filter by Artist", ["All"] + sorted(lyrics_db['Artist'].unique()))
year = st.sidebar.selectbox("Filter by Year", ["All"] + sorted(lyrics_db['Year'].unique()))
user_input = st.text_input("How are you feeling today?", "I feel joyful")
playlist_size = st.slider("Playlist Size", 3, 15, 5)
activity = st.selectbox("Activity Type", ['chill','study','party','focus'])

if st.button("Get My Playlist!"):
    emotion = get_emotion(user_input)
    playlist = recommend_playlist(user_input, lyrics_db, playlist_size, artist, year, activity, emotion)
    
    if playlist.empty:
        st.warning("No songs found with the selected filters. Try different filter options.")
    else:
        st.session_state.last_playlist = playlist
        st.session_state.history.extend(playlist['Song'].tolist())
        for idx,row in playlist.iterrows():
            st.write(f"**{row['Song']}** by *{row['Artist']}* | Year: {row['Year']}, Tempo: {row['tempo']}, Energy: {row['energy']}")
            yt_link = f"https://www.youtube.com/results?search_query={row['Song']}+{row['Artist']}"
            st.markdown(f'<a href="{yt_link}" target="_blank">▶ Play on YouTube</a>',unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            if col1.button(f"👍 Like {row['Song']}", key=f"up_{idx}"): st.session_state.ratings[row['Song']] = 1
            if col2.button(f"👎 Dislike {row['Song']}", key=f"down_{idx}"): st.session_state.ratings[row['Song']] = -1

st.header("Listening History")
st.write(", ".join(st.session_state.history[-20:]))

if not st.session_state.last_playlist.empty:
    st.header("Playlist Mood Analysis")
    show_mood_chart(st.session_state.last_playlist)
    show_tsne(st.session_state.last_playlist)

with st.expander("See Project Technical Summary 🚩"):
    st.markdown("""
    - Multi-modal mood & emotion detection
    - Genre/language filters & activity-aware playlisting
    - Spotify audio feature integration
    - User feedback and evolving recommendations
    - Advanced visualizations
    - Container-ready; CI/CD enabled
    """)
