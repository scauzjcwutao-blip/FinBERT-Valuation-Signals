"""
Project: NLP-Driven Asset Pricing Factors (Fixed Version)
"""
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoConfig
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
class Config:
    MODEL_NAME = "ProsusAI/finbert"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and verify label mapping
config = AutoConfig.from_pretrained(Config.MODEL_NAME)
print(f"Model label mapping: {config.id2label}")

tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(Config.MODEL_NAME)
model.to(Config.DEVICE).eval()

def get_sentiment_score(text_list):
    """Fixed version with error handling and device management."""
    # Input validation
    clean_texts = [text.replace('\n', ' ').strip() if pd.notna(text) else "" 
                   for text in text_list]
    
    inputs = tokenizer(clean_texts, return_tensors="pt", padding=True, 
                      truncation=True, max_length=Config.MAX_LENGTH)
    inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        
    return probs.cpu().numpy()

# --- Rest of the code remains similar but uses fixed functions ---
