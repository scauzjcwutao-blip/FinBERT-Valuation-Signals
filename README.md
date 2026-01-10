# FinBERT-Valuation-Signals
A pipeline extracting valuation factors from financial texts using FinBERT. Demo for Research Fellow application (TUM).
# NLP-Driven Asset Pricing Factors 📈🤖

## Overview
This repository demonstrates a robust pipeline for extracting **"Soft Information"** from unstructured financial texts (e.g., 10-K filings, Earnings Call transcripts) to construct quantitative factors for **Asset Pricing** models.

Using **FinBERT** (a BERT model pre-trained on financial text), this project transforms qualitative narratives into quantitative signals (Sentiment, ESG risks, Uncertainty), which can be used to explain cross-sectional return anomalies or valuation puzzles.

## Key Features
- **Domain-Specific Logic**: Utilizes `ProsusAI/finbert` for financial context awareness (vs. generic NLP).
- **Factor Construction**: Implements the standard log-odds ratio method `ln((1+Pos)/(1+Neg))` to create unbounded sentiment factors suitable for regression analysis.
- **Robust Engineering**: Includes GPU acceleration support, batch processing readiness, and rigorous data cleaning.

## Pipeline Architecture
1. **Input**: Raw text data (simulated 10-K excerpts).
2. **Preprocessing**: Tokenization and cleaning (handling specialized financial formats).
3. **Inference**: GPU-accelerated sentiment scoring via Transformer models.
4. **Factor Generation**: Converting probabilities into tradeable signals (Alpha).
5. **Visualization**: Visualizing the distribution of sentiment factors across a cross-section of assets.

## Future Research Directions (Proposed)
- **Scalability**: Scaling the pipeline to the full WRDS/EDGAR universe (1995-2025).
- **GenAI Integration**: Integrating LLMs (e.g., Llama 3) to extract more nuanced "Narrative Economics" features beyond simple sentiment.
- **Valuation Anomalies**: Testing the explanatory power of these text-based factors on the "Intangible Value" puzzle.

## Author
**Tao Wu**  
*M.Sc. Business Law & Taxation (University of Mannheim)*  
*Associate Professor (Guangzhou)*  
*Focus: Empirical Asset Pricing, NLP, Digital Assets*
