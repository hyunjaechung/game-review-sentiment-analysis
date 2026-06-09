import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Page config
st.set_page_config(
    page_title="Video Game Review Analytics",
    page_icon="🎮",
    layout="wide"
)

# Title
st.title("🎮 LLM-Powered Video Game Review Analytics")
st.markdown("**Author:** Hyunjae Chung")
st.markdown("Sentiment analysis of Amazon Video Game Reviews using GPT-4o-mini")
st.markdown("---")

# Load data
@st.cache
def load_data():
    df = pd.read_csv('./sentiment_results.csv')
    # Add star-rating-based sentiment for validation
    def rating_to_sentiment(stars):
        if stars >= 4:
            return 'Positive'
        elif stars == 3:
            return 'Neutral'
        else:
            return 'Negative'
    df['rating_sentiment'] = df['overall'].apply(rating_to_sentiment)
    df['match'] = df['sentiment'] == df['rating_sentiment']
    return df

df = load_data()

# ── SIDEBAR: Interactive Filters ─────────────────────
st.sidebar.header("🔧 Filters")

# Year range slider
min_year = int(df['year'].min())
max_year = int(df['year'].max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Sentiment multiselect
all_sentiments = ['Positive', 'Negative', 'Neutral']
selected_sentiments = st.sidebar.multiselect(
    "Select Sentiments",
    options=all_sentiments,
    default=all_sentiments
)

# Apply filters
filtered_df = df[
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1]) &
    (df['sentiment'].isin(selected_sentiments))
]

st.sidebar.markdown(f"**Filtered reviews:** {len(filtered_df)}")

# ── Section 1: Overview ──────────────────────────────
st.header("📊 Overall Sentiment Distribution")
st.caption(f"Showing data for years {year_range[0]} to {year_range[1]}")

col1, col2, col3 = st.columns(3)

positive = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
negative = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
neutral  = len(filtered_df[filtered_df['sentiment'] == 'Neutral'])
total    = max(positive + negative + neutral, 1)  # avoid divide by zero

with col1:
    st.metric("Positive", f"{positive}", f"{positive/total*100:.1f}%")
with col2:
    st.metric("Negative", f"{negative}", f"{negative/total*100:.1f}%")
with col3:
    st.metric("Neutral", f"{neutral}", f"{neutral/total*100:.1f}%")

st.markdown("---")

# ── Section 2: Sentiment Trend ───────────────────────
st.header("📈 Sentiment Trend by Year")

trend_df = filtered_df[filtered_df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

if len(trend_df) > 0:
    yearly = trend_df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
    yearly_pct = yearly.div(yearly.sum(axis=1), axis=0) * 100

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}

    for sentiment in selected_sentiments:
        if sentiment in yearly_pct.columns:
            ax1.plot(
                yearly_pct.index,
                yearly_pct[sentiment],
                marker='o',
                linewidth=2.5,
                label=sentiment,
                color=colors.get(sentiment, '#333333')
            )

    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Sentiment Trend', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
else:
    st.warning("No data for the selected filters.")

st.markdown("---")

# ── Section 3: Validation ────────────────────────────
st.header("🔍 Validation: Star Rating vs GPT Sentiment")
st.markdown("Star ratings are mapped to sentiment as follows: **4-5 stars = Positive, 3 stars = Neutral, 1-2 stars = Negative.**")

agreement_rate = filtered_df['match'].mean() * 100 if len(filtered_df) > 0 else 0

col4, col5 = st.columns(2)

with col4:
    st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
    st.write(f"Matching cases: {filtered_df['match'].sum()}")
    st.write(f"Disagreeing cases: {(~filtered_df['match']).sum()}")

with col5:
    try:
        heatmap_img = Image.open('./agreement_heatmap.png')
        st.image(heatmap_img, caption='Star Rating vs GPT Sentiment Heatmap')
    except:
        st.write("Heatmap image not found.")

st.info("The most common disagreement: 3-star reviews are often classified as Negative by GPT, suggesting mid-range ratings frequently reflect dissatisfaction rather than true neutrality.")

st.markdown("---")

# ── Section 4: Review Explorer (interactive table) ───
st.header("🔎 Review Explorer")
st.markdown("Browse individual reviews based on the filters in the sidebar.")

explorer_df = filtered_df[['overall', 'year', 'sentiment', 'reviewText']].copy()
explorer_df.columns = ['Stars', 'Year', 'GPT Sentiment', 'Review Text']
st.dataframe(explorer_df)

st.markdown("---")
st.caption("Data source: Amazon Video Game Reviews | Model: GPT-4o-mini | Sample: 999 reviews (2010-2018)")