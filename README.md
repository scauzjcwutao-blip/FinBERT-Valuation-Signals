# FinBERT-ESG-Reporting-Signals
A pipeline extracting transparency and sentiment signals from financial and non-financial texts using FinBERT. Demo for PhD Candidate application (TalTech).

# AI-Driven Financial & Non-Financial Reporting Analysis 📊🤖🌱

## Overview
This repository demonstrates a robust pipeline for extracting **"Soft Information"** from unstructured corporate disclosures (e.g., 10-K filings, ESG/Sustainability Reports, Earnings Call transcripts). 

Aligned with the critical debates on **Artificial Intelligence, ESG, and Sustainable Development**, this project utilizes **FinBERT** (a BERT model pre-trained on financial text) to transform qualitative narratives into quantitative signals. These signals can be used to assess corporate transparency, evaluate non-financial reporting quality, and detect potential "greenwashing" in corporate communications.

## Key Features
- **Domain-Specific Logic**: Utilizes `ProsusAI/finbert` for financial and corporate context awareness (outperforming generic NLP models in corporate reporting analysis).
- **Signal Construction**: Implements the standard log-odds ratio method `ln((1+Pos)/(1+Neg))` to create unbounded sentiment and risk factors suitable for empirical research in corporate governance.
- **Robust Engineering**: Includes GPU acceleration support, batch processing readiness, and rigorous data cleaning for complex corporate disclosure formats.

## Pipeline Architecture
1. **Input**: Raw text data (simulated excerpts from Annual Reports and ESG Disclosures).
2. **Preprocessing**: Tokenization and cleaning (handling specialized financial and regulatory formats).
3. **Inference**: GPU-accelerated sentiment and risk scoring via Transformer models.
4. **Signal Generation**: Converting textual probabilities into analytical metrics for reporting quality and ESG commitment.
5. **Visualization**: Visualizing the distribution of transparency and sentiment factors across different corporate entities.

## Future Research Directions (Proposed for PhD Project)
- **Cross-Jurisdictional Scalability**: Expanding the pipeline to analyze reporting frameworks across the EU (e.g., CSRD/ESRS) and global standards, assessing how regulatory environments shape AI-driven disclosures.
- **GenAI & RAG Integration**: Integrating advanced LLMs (e.g., Llama 3) with Retrieval-Augmented Generation (RAG) to extract more nuanced "Narrative Economics" features, specifically targeting the authenticity of sustainable development claims (anti-greenwashing).
- **Critical Debates on AI in Reporting**: Investigating the ethical implications, algorithmic accountability, and stakeholder trust when AI is utilized in generating and auditing financial and non-financial reports.

## Author
**Tao Wu**  
*M.Sc. Business Law & Taxation (University of Mannheim)*  
*Associate Professor*  
*Focus: AI in Corporate Reporting, ESG Disclosures, Sustainable Development, Digital Transformation*  
🔗 [LinkedIn: Insights on ESG & AI](https://www.linkedin.com/posts/tao-wu-a5a80a247_esg-artificialintelligence-sustainabledevelopment-activity-7420331840277807104-Enx-?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD0qmWYBmlakujovDrInUmI44Tt4GHND8rA)
