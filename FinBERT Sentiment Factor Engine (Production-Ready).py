"""
Project: NLP-Driven Asset Pricing Factors
Module: Sentiment Extraction Engine (Production Ready)
Author: [Your Name]
Date: 2026-01-11

Description: 
    Robust implementation of FinBERT inference for asset pricing research.
    Features:
    - Dynamic label mapping (Model-Agnostic)
    - Batch processing for VRAM safety
    - Log-odds ratio factor construction with numerical stability
    - Built-in unit testing suite
"""

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoConfig
from torch.nn.functional import softmax
from tqdm import tqdm  # Progress bar for UX
import warnings

# Suppress minor huggingface warnings for cleaner demo output
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & SETUP ---
class Config:
    MODEL_NAME = "ProsusAI/finbert"
    MAX_LENGTH = 128      # Optimized for demo speed (Use 512 for production)
    BATCH_SIZE = 32       # Adjustable based on GPU VRAM
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPSILON = 1e-8        # Numerical stability constant

class SentimentEngine:
    """
    Encapsulates the FinBERT model with robust inference pipelines.
    """
    def __init__(self):
        print(f"Initializing model on {Config.DEVICE}...")
        
        # 1. Load Configuration for Dynamic Label Mapping
        # This prevents hardcoding indices (e.g., assuming 0 is always Positive)
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.id2label = self.config.id2label
        
        # Dynamically find indices for 'positive' and 'negative' labels
        try:
            self.pos_idx = next(k for k, v in self.id2label.items() if 'positive' in v.lower())
            self.neg_idx = next(k for k, v in self.id2label.items() if 'negative' in v.lower())
            print(f"Label Mapping Found -> Positive: {self.pos_idx}, Negative: {self.neg_idx}")
        except StopIteration:
            raise ValueError("Error: Could not automatically determine positive/negative labels from model config.")

        # 2. Load Model & Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(Config.MODEL_NAME)
        self.model.to(Config.DEVICE).eval() # Set to evaluation mode for inference

    def _process_batch(self, texts):
        """Internal helper to process a single batch of text."""
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=Config.MAX_LENGTH
        )
        # Move inputs to the active device (GPU/CPU)
        inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply Softmax to get probabilities (Sum=1)
            probs = softmax(outputs.logits, dim=1)
            
        return probs.cpu().numpy()

    def get_scores(self, text_list):
        """
        Main inference pipeline with batching, progress tracking, and error handling.
        """
        # 1. Input Validation & Cleaning
        # Handle NaNs and force string type to prevent crashes
        clean_texts = [str(t).replace('\n', ' ').strip() if pd.notna(t) else "" 
                       for t in text_list]
        
        all_probs = []
        
        # 2. Batched Inference Loop
        # Process data in chunks to avoid Out-Of-Memory (OOM) errors
        num_batches = (len(clean_texts) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        
        # tqdm provides a progress bar, essential for long-running jobs
        for i in tqdm(range(0, len(clean_texts), Config.BATCH_SIZE), 
                     desc="Running Inference", total=num_batches):
            
            batch_texts = clean_texts[i : i + Config.BATCH_SIZE]
            
            # Skip completely empty batches if any
            if not batch_texts: continue
                
            batch_probs = self._process_batch(batch_texts)
            all_probs.append(batch_probs)
            
        if not all_probs:
            return np.array([])
            
        return np.vstack(all_probs)

# --- 2. FACTOR CONSTRUCTION LOGIC ---
def construct_factor(df, engine, text_col='Text'):
    """
    Applies the engine to a DataFrame and calculates the Sentiment Factor.
    """
    scores = engine.get_scores(df[text_col].tolist())
    
    # Map scores to columns dynamically using engine's indices
    df['Prob_Pos'] = scores[:, engine.pos_idx]
    df['Prob_Neg'] = scores[:, engine.neg_idx]
    
    # Financial Logic: Log-Odds Ratio
    # Formula: ln( (Pos + eps) / (Neg + eps) )
    # This standardizes the sentiment score for regression analysis
    df['Sentiment_Factor'] = np.log(
        (df['Prob_Pos'] + Config.EPSILON) / 
        (df['Prob_Neg'] + Config.EPSILON)
    )
    
    # Winsorize/Clip extreme values (Standard Quant Practice)
    # Prevents outliers from skewing the asset pricing tests
    df['Sentiment_Factor'] = df['Sentiment_Factor'].clip(-10, 10)
    
    return df

# --- 3. UNIT TESTING SUITE ---
def test_engine():
    """
    Unit Test: Verifies pipeline integrity and logical correctness.
    """
    print("\n[Self-Check] Running unit tests...")
    
    # 1. Initialize
    engine = SentimentEngine()
    
    # 2. Test Cases (Positive, Negative, Edge Case)
    test_texts = [
        "The company reported amazing profits and record growth.",  # Should be Positive
        "We are facing bankruptcy and huge losses.",                # Should be Negative
        "",                                                       # Empty String
        np.nan                                                    # NaN
    ]
    
    # 3. Execution
    scores = engine.get_scores(test_texts)
    
    # 4. Assertions (The "Double Check")
    # Check Shape: Should match input length (4 samples) and model classes (3 classes)
    assert scores.shape == (4, 3), f"Shape Mismatch! Expected (4, 3), got {scores.shape}"
    
    # Check Logic: 
    # Sample 0 (Amazing profits) -> Pos score should be highest
    assert scores[0, engine.pos_idx] > scores[0, engine.neg_idx], "Logic Fail: Positive text got low score!"
    
    # Sample 1 (Bankruptcy) -> Neg score should be highest
    assert scores[1, engine.neg_idx] > scores[1, engine.pos_idx], "Logic Fail: Negative text got low score!"
    
    print("✓ All tests passed: Shape is correct and Sentiment Logic holds.\n")
    return engine # Return engine to reuse it and save load time

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # Step 1: Run Unit Tests
    try:
        engine = test_engine()
    except Exception as e:
        print(f"CRITICAL: Unit tests failed. {e}")
        exit()

    # Step 2: Run Demo on Financial Data
    print("Starting Demo Execution...")
    
    # Create Dummy Financial Data
    data = {
        'Ticker': ['AAPL', 'TSLA', 'XOM', 'JPM', 'Unknown'],
        'Text': [
            "Apple reports record-breaking revenue and strong iPhone demand.",  # Clearly Positive
            "Tesla recalls vehicles due to software glitch; stock tumbles.",    # Clearly Negative
            "Oil prices stabilize as geopolitical tensions ease slightly.",     # Neutral/Mixed
            np.nan,                                                           # Edge Case: NaN
            "   "                                                             # Edge Case: Empty
        ]
    }
    df_test = pd.DataFrame(data)

    try:
        df_result = construct_factor(df_test, engine)
        
        print("\n--- Final Factor Data (Preview) ---")
        # Format output for cleaner display
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(df_result[['Ticker', 'Prob_Pos', 'Prob_Neg', 'Sentiment_Factor']])
        
        print("\nSuccess: Pipeline is robust and ready for production.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
