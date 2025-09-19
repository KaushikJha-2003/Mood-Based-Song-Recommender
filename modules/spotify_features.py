import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Only initialize if credentials are available
if SPOTIFY_ID and SPOTIFY_SECRET:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_ID, client_secret=SPOTIFY_SECRET))
else:
    sp = None

def add_audio_features(df):
    features = []
    for idx, row in df.iterrows():
        if sp is None:
            # Return dummy values if Spotify is not available
            features.append({'danceability': 0.5, 'energy': 0.5, 'tempo': 120})
            continue
            
        try:
            track = f"{row['Song']} {row['Artist']}"
            results = sp.search(q=track, type="track", limit=1)
            if results['tracks']['items']:
                track_id = results['tracks']['items'][0]['id']
                afeat = sp.audio_features(track_id)[0]
                if afeat:
                    features.append({'danceability': afeat['danceability'], 'energy': afeat['energy'], 'tempo': afeat['tempo']})
                else:
                    features.append({'danceability': None, 'energy': None, 'tempo': None})
            else:
                features.append({'danceability': None, 'energy': None, 'tempo': None})
        except Exception as e:
            print(f"Spotify API error: {e}")
            features.append({'danceability': None, 'energy': None, 'tempo': None})
    
    for feat in ['danceability','energy','tempo']:
        df[feat] = [f[feat] for f in features]
    return df
