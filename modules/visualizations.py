import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import streamlit as st

def show_mood_chart(playlist):
    mood_counts = playlist['emotion'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(mood_counts.index, mood_counts.values)
    st.pyplot(fig)

def show_tsne(playlist):
    model_embs = playlist['Lyrics']
    if isinstance(model_embs.iloc[0], np.ndarray): embeddings = np.vstack(model_embs)
    else: embeddings = np.array([np.random.rand(384) for _ in range(len(playlist))])
    tsne_results = TSNE(n_components=2, perplexity=min(30,len(embeddings))-1).fit_transform(embeddings)
    plt.figure(figsize=(7,5))
    plt.scatter(tsne_results[:,0], tsne_results[:,1], c='blue')
    for i, name in enumerate(playlist['Song']):
        plt.text(tsne_results[i,0], tsne_results[i,1], name)
    st.pyplot(plt.gcf())
