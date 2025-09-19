from transformers import pipeline
import torch

emotion_model = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student", framework="pt")

def get_emotion(text):
    res = emotion_model(text)
    return res[0]['label']
