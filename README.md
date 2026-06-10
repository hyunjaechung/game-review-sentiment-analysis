# LLM-Powered Video Game Review Sentiment Analysis

Sentiment analysis of Amazon video game reviews using GPT-4o-mini, with a star-rating validation layer, yearly trend analysis, a BigQuery data pipeline, and an interactive Streamlit dashboard.

## Overview

This project classifies the sentiment of Amazon video game reviews (2010–2018) using a large language model, then validates those classifications against the reviewers' own star ratings. The goal is to demonstrate an end-to-end workflow: raw data cleaning, LLM-based classification, validation, cloud data warehousing, and interactive visualization.

## Key Results

- **999 reviews** classified with GPT-4o-mini (stratified sample across 2010–2018)
- **Sentiment distribution:** 668 Positive (66.9%), 209 Negative (20.9%), 116 Neutral (11.6%), 6 Mixed
- **83.7% agreement** between GPT sentiment and star-rating-derived sentiment
- **Most common disagreement:** 3-star reviews are frequently classified as Negative by GPT, suggesting mid-range ratings often reflect dissatisfaction rather than true neutrality

## Sentiment Trend (2010–2018)

![Sentiment Trend](sentiment_trend.png)

Positive sentiment peaked around 2013 (~73%) and gradually declined toward 2017–2018 (~60%), while negative sentiment stayed relatively stable at 17–23%.

## Validation: Star Rating vs GPT Sentiment

![Agreement Heatmap](agreement_heatmap.png)

Star ratings are mapped to sentiment (4–5 stars = Positive, 3 stars = Neutral, 1–2 stars = Negative) and compared against GPT's classifications. The strong diagonal confirms high agreement, while off-diagonal cells reveal where the LLM captures nuance that a single star rating cannot.

## Tech Stack

- **Python** (pandas, matplotlib, numpy)
- **OpenAI GPT-4o-mini** for sentiment classification
- **LangChain** for a reusable prompt → model → parser pipeline
- **Google BigQuery** for cloud data warehousing and SQL querying
- **Streamlit** for the interactive dashboard

## Pipeline

1. **Data cleaning** – Load the raw Amazon review dataset, drop missing/short reviews, parse dates, and filter to 2010 onward.
2. **Sentiment classification** – Use GPT-4o-mini to classify a stratified sample of 999 reviews (Positive / Negative / Neutral).
3. **Trend analysis** – Compute yearly sentiment percentages and plot the trend over time.
4. **Validation** – Map star ratings to sentiment and measure agreement with GPT via a confusion-matrix heatmap.
5. **LangChain refactor** – Re-implement the classification logic as a reusable LangChain chain.
6. **BigQuery integration** – Upload results to BigQuery and query them back from Python as a pandas DataFrame, completing the end-to-end pipeline (local analysis → cloud warehouse → SQL → back into Python).

## Repository Structure

```
LLM Project.ipynb       # Full analysis notebook (Steps 1–12)
dashboard.py            # Interactive Streamlit dashboard
sentiment_results.csv   # 999 classified reviews (analysis output)
sentiment_trend.png     # Yearly sentiment trend chart
agreement_heatmap.png   # Star rating vs GPT sentiment heatmap
requirements.txt        # Python dependencies
```

## Running Locally

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

To re-run the notebook's LLM classification or BigQuery steps, you'll need your own OpenAI API key (in a `.env` file as `OPENAI_API_KEY`) and a Google Cloud service account key. The notebook's API calls are commented out by default so it runs reproducibly from the saved results.

## Data Source

Amazon Video Game Reviews dataset (2010–2018). The raw dataset (~497K reviews) is not included in this repository due to size; the cleaned and classified results are provided in `sentiment_results.csv`.

## Notes

- GPT occasionally returns "Mixed" despite being prompted for three categories, reflecting genuinely ambivalent reviews.
- Python 3.10 is used here; Google libraries will require 3.11+ after October 2026.
