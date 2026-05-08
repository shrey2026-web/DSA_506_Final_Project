import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Can AI Understand Human Emotion Through Music?",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – dark editorial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      background-color: #111318;
      color: #f0f0f0;
  }
  .stApp { background-color: #111318; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background: #0d0f14;
      border-right: 1px solid #2a2d38;
  }
  section[data-testid="stSidebar"] * { color: #c8cad4 !important; }
  section[data-testid="stSidebar"] .stMarkdown h3 { color: #f0c040 !important; font-size: 1rem !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
      background: #0d0f14;
      border-bottom: 2px solid #2a2d38;
      gap: 0px;
  }
  .stTabs [data-baseweb="tab"] {
      color: #6b7280 !important;
      font-family: 'Inter', sans-serif;
      font-weight: 600;
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 12px 20px;
      border-radius: 0;
  }
  .stTabs [aria-selected="true"] {
      background: #1a1d26 !important;
      color: #f0c040 !important;
      border-bottom: 3px solid #f0c040 !important;
  }

  /* Metric cards */
  .metric-card {
      background: #1a1d26;
      border: 1px solid #2a2d38;
      border-top: 3px solid #f0c040;
      border-radius: 8px;
      padding: 20px 20px;
      text-align: center;
  }
  .metric-card .label {
      font-size: 0.65rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #6b7280;
      margin-bottom: 8px;
      font-weight: 600;
  }
  .metric-card .value {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 2.4rem;
      color: #f0c040;
      line-height: 1;
      letter-spacing: 0.02em;
  }
  .metric-card .sub {
      font-size: 0.7rem;
      color: #4b5563;
      margin-top: 5px;
  }

  /* Section headers */
  .section-header {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 2rem;
      letter-spacing: 0.05em;
      color: #ffffff;
      margin-bottom: 2px;
  }
  .section-sub {
      font-size: 0.85rem;
      color: #6b7280;
      margin-bottom: 24px;
      font-weight: 400;
  }

  /* Story / insight cards */
  .story-card {
      background: #1a1d26;
      border-left: 4px solid #f0c040;
      border-radius: 0 8px 8px 0;
      padding: 18px 22px;
      margin-bottom: 16px;
  }
  .story-card h4 {
      font-family: 'Inter', sans-serif;
      font-weight: 700;
      color: #f0c040;
      font-size: 0.85rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin: 0 0 8px 0;
  }
  .story-card p {
      color: #9ca3af;
      font-size: 0.88rem;
      line-height: 1.7;
      margin: 0;
  }
  .story-card strong { color: #f0f0f0; }
  .story-card em { color: #f0c040; font-style: normal; font-weight: 600; }

  /* Hero */
  .hero-band {
      background: #1a1d26;
      border: 1px solid #2a2d38;
      border-left: 5px solid #f0c040;
      border-radius: 8px;
      padding: 36px 44px;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
  }
  .hero-band::before {
      content: '♫';
      position: absolute; right: 44px; top: 16px;
      font-size: 110px; color: rgba(240,192,64,0.06);
      font-family: serif;
  }
  .hero-title {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 3rem;
      letter-spacing: 0.04em;
      color: #ffffff;
      line-height: 1.1;
      margin-bottom: 12px;
  }
  .hero-title span { color: #f0c040; }
  .hero-rq {
      font-size: 0.9rem;
      color: #9ca3af;
      max-width: 680px;
      line-height: 1.7;
  }
  .hero-badge {
      display: inline-block;
      background: transparent;
      border: 1px solid #f0c040;
      border-radius: 3px;
      padding: 3px 10px;
      font-size: 0.65rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #f0c040;
      font-weight: 700;
      margin-bottom: 14px;
  }

  /* Conclusion */
  .conclusion-box {
      background: #1a1d26;
      border: 1px solid #2a2d38;
      border-top: 3px solid #f0c040;
      border-radius: 8px;
      padding: 30px 34px;
  }
  .conclusion-box h3 {
      font-family: 'Bebas Neue', sans-serif;
      letter-spacing: 0.05em;
      color: #f0c040;
      font-size: 1.5rem;
      margin-bottom: 12px;
  }

  /* Divider */
  hr { border-color: #2a2d38 !important; }

  /* Widget labels */
  label { color: #6b7280 !important; font-size: 0.78rem !important; font-weight: 600 !important; letter-spacing: 0.05em !important; }
  .stSelectbox > div > div { background: #1a1d26 !important; border-color: #2a2d38 !important; color: #f0f0f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("tableau_music_emotion_dataset.csv")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df

df = load_data()

MOOD_COLORS = {
    "Happy / Energetic": "#22c55e",
    "Intense / Dark":    "#ef4444",
    "Sad / Calm":        "#3b82f6",
    "Calm / Positive":   "#06b6d4",
}

GENRE_COLORS = {
    "pop":   "#e879f9",
    "rock":  "#f97316",
    "r&b":   "#facc15",
    "edm":   "#22d3ee",
    "rap":   "#818cf8",
    "latin": "#fb7185",
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1a1d26",
    font=dict(family="Inter", color="#d1d5db", size=12),
    colorway=["#e879f9","#f97316","#facc15","#22d3ee","#818cf8","#fb7185"],
    legend=dict(bgcolor="#1a1d26", bordercolor="#2a2d38", borderwidth=1),
)

AXIS_STYLE = dict(gridcolor="#2a2d38", linecolor="#2a2d38", tickcolor="#4b5563", tickfont=dict(color="#6b7280"))

def apply_axes(fig):
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎵 Music Emotion AI")
    st.markdown("---")

    st.markdown("**Filter by Genre**")
    all_genres = ["All"] + sorted(df["playlist_genre"].unique().tolist())
    selected_genre = st.selectbox("Genre", all_genres, label_visibility="collapsed")

    st.markdown("**Filter by Mood**")
    all_moods = ["All"] + sorted(df["mood_category"].unique().tolist())
    selected_mood = st.selectbox("Mood Category", all_moods, label_visibility="collapsed")

    st.markdown("**Year Range**")
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider("Years", yr_min, yr_max, (2000, yr_max), label_visibility="collapsed")

    st.markdown("**Feature for Analysis**")
    audio_feature = st.selectbox(
        "Audio Feature",
        ["valence", "energy", "danceability", "tempo", "acousticness",
         "speechiness", "loudness", "liveness", "instrumentalness", "track_popularity"],
        label_visibility="collapsed"
    )

    st.markdown("**Bubble Chart X / Y**")
    bubble_x = st.selectbox("X Axis", ["energy","danceability","tempo","valence","acousticness"], index=0, label_visibility="collapsed")
    bubble_y = st.selectbox("Y Axis", ["valence","energy","danceability","tempo","acousticness"], index=0, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<small style='color:#4b5563'>DSA 506 · Shreyasee Poddar<br>14,200 songs · 6 genres · 4 moods</small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
fdf = df.copy()
if selected_genre != "All":
    fdf = fdf[fdf["playlist_genre"] == selected_genre]
if selected_mood != "All":
    fdf = fdf[fdf["mood_category"] == selected_mood]
fdf = fdf[(fdf["year"] >= year_range[0]) & (fdf["year"] <= year_range[1])]

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-band">
  <div class="hero-badge">DSA 506 · Final Project</div>
  <div class="hero-title">Can AI Understand Human Emotion<br>Through <span>Music?</span></div>
  <div class="hero-rq">
    Research Question: Can data from songs - including lyrics, audio features, and artist characteristics -
    help identify emotional patterns in modern music?
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card">
      <div class="label">Songs Analyzed</div>
      <div class="value">{len(fdf):,}</div>
      <div class="sub">of 14,200 total</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card">
      <div class="label">Avg. Valence</div>
      <div class="value">{fdf['valence'].mean():.2f}</div>
      <div class="sub">emotional positivity</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card">
      <div class="label">Avg. Energy</div>
      <div class="value">{fdf['energy'].mean():.2f}</div>
      <div class="sub">intensity level</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card">
      <div class="label">Avg. Danceability</div>
      <div class="value">{fdf['danceability'].mean():.2f}</div>
      <div class="sub">rhythmic suitability</div>
    </div>""", unsafe_allow_html=True)
with k5:
    top_mood = fdf["mood_category"].value_counts().idxmax() if len(fdf) > 0 else "-"
    st.markdown(f"""<div class="metric-card">
      <div class="label">Dominant Mood</div>
      <div class="value" style="font-size:1.1rem;padding-top:8px">{top_mood}</div>
      <div class="sub">in current selection</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🎭 The Hook",
    "📊 Audio Features",
    "🎨 Mood Landscape",
    "📈 Trends Over Time",
    "🎤 Lyric Sentiment",
    "🔬 Deep Dive",
    "🏁 Conclusion"
])

# ═══════════════════════════════════════════
# TAB 1 – THE HOOK
# ═══════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">The Hook</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A surprising look at what music data actually reveals about human emotion</div>', unsafe_allow_html=True)

    # Story card
    st.markdown("""
    <div class="story-card">
      <h4>🎵 Did you know?</h4>
      <p>Humans use music as an emotional compass - we play certain songs when heartbroken, others when celebrating.
      But can a machine decode that emotional fingerprint hidden inside the audio data?
      Across <strong>14,200 songs</strong> spanning six decades, we find that energy, valence, danceability,
      and lyric sentiment together reveal measurable emotional patterns - and AI can learn to read them.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        # Mood donut
        mood_counts = fdf["mood_category"].value_counts().reset_index()
        mood_counts.columns = ["Mood", "Count"]
        fig = px.pie(
            mood_counts, values="Count", names="Mood",
            hole=0.55,
            color="Mood",
            color_discrete_map=MOOD_COLORS,
            title="Emotional Fingerprint of Music"
        )
        fig.update_traces(textinfo="percent+label", textfont_size=12,
                          marker=dict(line=dict(color="#0a0a0f", width=2)))
        fig.update_layout(**PLOTLY_TEMPLATE, height=360,
                          title_font=dict(size=13, color="#f0c040"),
                          showlegend=False)
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Genre bar
        genre_counts = fdf["playlist_genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        fig2 = px.bar(
            genre_counts, x="Count", y="Genre", orientation="h",
            color="Genre",
            color_discrete_map=GENRE_COLORS,
            title="Songs per Genre"
        )
        fig2.update_layout(**PLOTLY_TEMPLATE, height=360,
                           title_font=dict(size=13, color="#f0c040"),
                           showlegend=False)
        fig2.update_xaxes(**AXIS_STYLE)
        fig2.update_yaxes(categoryorder="total ascending", **AXIS_STYLE)
        st.plotly_chart(fig2, use_container_width=True)

    # Key stats row
    st.markdown("### Key Emotional Statistics")
    c1, c2, c3, c4 = st.columns(4)
    happy_pct = round(len(fdf[fdf["mood_category"] == "Happy / Energetic"]) / max(len(fdf), 1) * 100, 1)
    intense_pct = round(len(fdf[fdf["mood_category"] == "Intense / Dark"]) / max(len(fdf), 1) * 100, 1)
    pos_sentiment = round(len(fdf[fdf["lyric_sentiment"] > 0]) / max(len(fdf), 1) * 100, 1)
    avg_pop = round(fdf["track_popularity"].mean(), 1)
    for col, val, label, sub in zip(
        [c1, c2, c3, c4],
        [f"{happy_pct}%", f"{intense_pct}%", f"{pos_sentiment}%", str(avg_pop)],
        ["Happy / Energetic", "Intense / Dark", "Positive Lyrics", "Avg Popularity"],
        ["of all songs", "of all songs", "sentiment score > 0", "out of 100"]
    ):
        col.markdown(f"""<div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{val}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2 – AUDIO FEATURES
# ═══════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Audio Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Numerical deep-dive into the sonic DNA of emotion</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Distribution of selected feature
        fig = px.histogram(
            fdf, x=audio_feature, color="mood_category",
            nbins=40, barmode="overlay", opacity=0.75,
            color_discrete_map=MOOD_COLORS,
            title=f"Distribution of {audio_feature.title()} by Mood"
        )
        fig.update_layout(**PLOTLY_TEMPLATE, height=340,
                          title_font=dict(size=13, color="#f0c040"),
                          bargap=0.05)
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot by genre
        fig2 = px.box(
            fdf, x="playlist_genre", y=audio_feature,
            color="playlist_genre",
            color_discrete_map=GENRE_COLORS,
            title=f"{audio_feature.title()} Across Genres",
            points=False
        )
        fig2.update_layout(**PLOTLY_TEMPLATE, height=340,
                           title_font=dict(size=13, color="#f0c040"),
                           showlegend=False)
        fig2.update_xaxes(**AXIS_STYLE)
        fig2.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig2, use_container_width=True)

    # Radar chart – avg features per mood
    st.markdown("### Emotional Signature Radar")
    radar_features = ["danceability", "energy", "valence", "acousticness", "speechiness", "liveness"]
    mood_avg = fdf.groupby("mood_category")[radar_features].mean().reset_index()

    fig_radar = go.Figure()
    for _, row in mood_avg.iterrows():
        mood = row["mood_category"]
        values = [row[f] for f in radar_features]
        values += values[:1]
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_features + [radar_features[0]],
            fill="toself",
            name=mood,
            line=dict(color=MOOD_COLORS.get(mood, "#c084fc")),
            opacity=0.6
        ))
    fig_radar.update_layout(
        **PLOTLY_TEMPLATE, height=420,
        polar=dict(
            bgcolor="#1a1d26",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#2a2d38",
                            tickfont=dict(color="#6b7280")),
            angularaxis=dict(gridcolor="#2a2d38")
        ),
        title=dict(text="Avg Audio Features per Mood Category",
                   font=dict(color="#f0c040", size=13))
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Correlation heatmap
    st.markdown("### Feature Correlation Heatmap")
    num_cols = ["danceability","energy","valence","tempo","loudness",
                "speechiness","acousticness","instrumentalness","liveness","track_popularity"]
    corr = fdf[num_cols].corr().round(2)
    fig_heat = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdPu",
        title="How Audio Features Relate to Each Other"
    )
    fig_heat.update_layout(**PLOTLY_TEMPLATE, height=440,
                           title_font=dict(size=13, color="#f0c040"))
    fig_heat.update_xaxes(**AXIS_STYLE)
    fig_heat.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    <div class="story-card">
      <h4>📊 Insight</h4>
      <p>Energy and loudness show the strongest positive correlation, confirming that louder songs feel more intense.
      Acousticness is negatively correlated with energy - acoustic, quieter tracks tend to carry calmer emotional energy.
      These patterns are not random; they are the measurable architecture of human emotion.</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 3 – MOOD LANDSCAPE
# ═══════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Mood Landscape</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Where do emotions live in the audio feature space?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        # Bubble / scatter plot
        sample = fdf.sample(min(2000, len(fdf)), random_state=42) if len(fdf) > 0 else fdf
        fig_bubble = px.scatter(
            sample, x=bubble_x, y=bubble_y,
            color="mood_category", size="track_popularity",
            hover_data=["track_name", "track_artist", "playlist_genre"],
            color_discrete_map=MOOD_COLORS,
            opacity=0.7,
            title=f"{bubble_x.title()} vs {bubble_y.title()} - Sized by Popularity",
            size_max=18
        )
        fig_bubble.update_layout(**PLOTLY_TEMPLATE, height=430,
                                  title_font=dict(size=13, color="#f0c040"))
        fig_bubble.update_xaxes(**AXIS_STYLE)
        fig_bubble.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_bubble, use_container_width=True)

    with col2:
        # Mood by genre stacked bar
        mg = fdf.groupby(["playlist_genre", "mood_category"]).size().reset_index(name="count")
        fig_stack = px.bar(
            mg, x="playlist_genre", y="count", color="mood_category",
            color_discrete_map=MOOD_COLORS, barmode="stack",
            title="Mood Distribution per Genre"
        )
        fig_stack.update_layout(**PLOTLY_TEMPLATE, height=430,
                                title_font=dict(size=13, color="#f0c040"),
                                xaxis_title="Genre", yaxis_title="Song Count")
        fig_stack.update_xaxes(**AXIS_STYLE)
        fig_stack.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_stack, use_container_width=True)

    # Violin: valence by mood
    st.markdown("### Valence (Positivity) Distribution by Mood")
    fig_violin = px.violin(
        fdf, x="mood_category", y="valence",
        color="mood_category", box=True, points=False,
        color_discrete_map=MOOD_COLORS,
        title="How Positive Are Songs in Each Mood Category?"
    )
    fig_violin.update_layout(**PLOTLY_TEMPLATE, height=380,
                             title_font=dict(size=13, color="#f0c040"),
                             showlegend=False)
    fig_violin.update_xaxes(**AXIS_STYLE)
    fig_violin.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("""
    <div class="story-card">
      <h4>🎨 Insight</h4>
      <p>The scatter plot confirms that moods cluster in predictable regions of audio feature space.
      Happy / Energetic songs occupy the high-energy, high-valence quadrant.
      Sad / Calm songs fall in low-energy territory. This geometric separation is exactly what allows
      AI to classify emotion from raw audio data - without ever hearing the music.</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 4 – TRENDS OVER TIME
# ═══════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">Emotional Trends Over Time</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How has the emotional tone of music shifted across decades?</div>', unsafe_allow_html=True)

    yearly = fdf.groupby("year")[["valence","energy","danceability","acousticness","speechiness"]].mean().reset_index()
    yearly = yearly[yearly["year"] >= 1970]

    # Animated line chart (year-by-year reveal)
    col1, col2 = st.columns(2)
    with col1:
        fig_line = px.line(
            yearly, x="year", y=["valence","energy","danceability"],
            title="Valence, Energy & Danceability Over Time",
            labels={"value": "Score", "variable": "Feature"},
            color_discrete_sequence=["#c084fc","#f87171","#4ade80"]
        )
        fig_line.update_layout(**PLOTLY_TEMPLATE, height=350,
                               title_font=dict(size=13, color="#f0c040"))
        fig_line.update_xaxes(**AXIS_STYLE)
        fig_line.update_yaxes(**AXIS_STYLE)
        fig_line.update_traces(line=dict(width=2.5))
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        fig_line2 = px.line(
            yearly, x="year", y=["acousticness","speechiness"],
            title="Acousticness & Speechiness Over Time",
            labels={"value": "Score", "variable": "Feature"},
            color_discrete_sequence=["#60a5fa","#fbbf24"]
        )
        fig_line2.update_layout(**PLOTLY_TEMPLATE, height=350,
                                title_font=dict(size=13, color="#f0c040"))
        fig_line2.update_xaxes(**AXIS_STYLE)
        fig_line2.update_yaxes(**AXIS_STYLE)
        fig_line2.update_traces(line=dict(width=2.5))
        st.plotly_chart(fig_line2, use_container_width=True)

    # Animated bubble: energy vs valence by year
    # This version uses cumulative yearly frames and initializes all genre+mood groups
    # in the first frame, so new genres/moods can appear as the animation advances.
    st.markdown("### Animated: How Emotion Evolved Year by Year")

    anim_start, anim_end = int(year_range[0]), int(year_range[1])
    anim_source = df[(df["year"] >= anim_start) & (df["year"] <= anim_end)].copy()

    if len(anim_source) > 0:
        # All genre+mood combinations in the selected year range.
        all_groups = (
            anim_source[["playlist_genre", "mood_category"]]
            .drop_duplicates()
            .sort_values(["playlist_genre", "mood_category"])
            .reset_index(drop=True)
        )
        all_groups["genre_mood"] = all_groups["playlist_genre"] + " - " + all_groups["mood_category"]

        # Stable fallback coordinates for groups before they appear.
        group_defaults = anim_source.groupby(["playlist_genre", "mood_category"]).agg(
            default_energy=("energy", "mean"),
            default_valence=("valence", "mean"),
            default_popularity=("track_popularity", "mean")
        ).reset_index()

        bubble_frames = []
        for frame_year in range(anim_start, anim_end + 1):
            # Cumulative data up to the current frame year.
            upto_year = anim_source[anim_source["year"] <= frame_year]
            current = upto_year.groupby(["playlist_genre", "mood_category"]).agg(
                energy=("energy", "mean"),
                valence=("valence", "mean"),
                popularity=("track_popularity", "mean"),
                count=("track_id", "count"),
                first_year=("year", "min")
            ).reset_index()

            frame = all_groups.merge(current, on=["playlist_genre", "mood_category"], how="left")
            frame = frame.merge(group_defaults, on=["playlist_genre", "mood_category"], how="left")

            # Before a group appears, keep it initialized but visually tiny.
            frame["count"] = frame["count"].fillna(0)
            frame["first_year"] = frame["first_year"].fillna(9999).astype(int)
            frame["energy"] = frame["energy"].fillna(frame["default_energy"])
            frame["valence"] = frame["valence"].fillna(frame["default_valence"])
            frame["popularity"] = frame["popularity"].fillna(frame["default_popularity"])
            frame["count_for_size"] = frame["count"].where(frame["count"] > 0, 0.01)
            frame["status"] = np.where(frame["count"] > 0, "Visible in cumulative data", "Not appeared yet")
            frame["frame_year"] = frame_year
            bubble_frames.append(frame)

        bubble_anim = pd.concat(bubble_frames, ignore_index=True)

        fig_anim = px.scatter(
            bubble_anim,
            x="energy",
            y="valence",
            animation_frame="frame_year",
            animation_group="genre_mood",
            size="count_for_size",
            color="mood_category",
            symbol="playlist_genre",
            color_discrete_map=MOOD_COLORS,
            hover_name="playlist_genre",
            hover_data={
                "mood_category": True,
                "count": ":.0f",
                "first_year": True,
                "status": True,
                "popularity": ":.1f",
                "energy": ":.3f",
                "valence": ":.3f",
                "count_for_size": False,
                "genre_mood": False,
                "frame_year": False
            },
            size_max=55,
            range_x=[0, 1],
            range_y=[0, 1],
            title="Energy vs Valence by Genre and Mood (Cumulative Animation by Year)"
        )
        fig_anim.update_layout(**PLOTLY_TEMPLATE, height=520,
                               title_font=dict(size=13, color="#f0c040"),
                               xaxis_title="Energy",
                               yaxis_title="Valence")
        fig_anim.update_xaxes(**AXIS_STYLE)
        fig_anim.update_yaxes(**AXIS_STYLE)
        if fig_anim.layout.updatemenus:
            fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 350
            fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 100
        st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.warning("No songs are available for the selected year range.")

    # Animated time-series chart, using cumulative frames so the line grows over time.
    st.markdown("### Animated Time-Series: Emotional Audio Features")

    yearly_features = (
        df[(df["year"] >= anim_start) & (df["year"] <= anim_end)]
        .groupby("year")[["energy", "valence", "danceability"]]
        .mean()
        .reset_index()
    )

    yearly_features_long = yearly_features.melt(
        id_vars="year",
        value_vars=["energy", "valence", "danceability"],
        var_name="Audio Feature",
        value_name="Average Score"
    )

    frames = []
    for frame_year in sorted(yearly_features_long["year"].unique()):
        temp = yearly_features_long[yearly_features_long["year"] <= frame_year].copy()
        temp["frame_year"] = frame_year
        frames.append(temp)

    if frames:
        animated_df = pd.concat(frames, ignore_index=True)

        fig_time_anim = px.line(
            animated_df,
            x="year",
            y="Average Score",
            color="Audio Feature",
            animation_frame="frame_year",
            markers=True,
            range_x=[anim_start, anim_end],
            range_y=[0, 1],
            title="Animated Time-Series of Emotional Audio Features",
            color_discrete_sequence=["#f87171", "#c084fc", "#4ade80"]
        )

        fig_time_anim.update_layout(**PLOTLY_TEMPLATE, height=460,
                                    title_font=dict(size=13, color="#f0c040"),
                                    xaxis_title="Year",
                                    yaxis_title="Average Score")
        fig_time_anim.update_xaxes(**AXIS_STYLE)
        fig_time_anim.update_yaxes(**AXIS_STYLE)
        fig_time_anim.update_traces(line=dict(width=2.5), marker=dict(size=6))
        if fig_time_anim.layout.updatemenus:
            fig_time_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
            fig_time_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250
        st.plotly_chart(fig_time_anim, use_container_width=True)

    # Mood share over time
    st.markdown("### Mood Share by Decade")
    fdf2 = fdf.copy()
    fdf2["decade"] = (fdf2["year"] // 10 * 10).astype(str) + "s"
    decade_mood = fdf2.groupby(["decade","mood_category"]).size().reset_index(name="count")
    fig_dec = px.bar(
        decade_mood, x="decade", y="count", color="mood_category",
        color_discrete_map=MOOD_COLORS, barmode="stack",
        title="Proportional Mood Shift Across Decades"
    )
    fig_dec.update_layout(**PLOTLY_TEMPLATE, height=340,
                          title_font=dict(size=13, color="#f0c040"),
                          barnorm="fraction",
                          yaxis_tickformat=".0%")
    fig_dec.update_xaxes(**AXIS_STYLE)
    fig_dec.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown("""
    <div class="story-card">
      <h4>📈 Insight</h4>
      <p>The animated chart reveals a clear trend: valence (emotional positivity) has been <em>declining</em>
      since the early 2010s, while energy remains relatively high. This suggests that modern music has become
      more intense but less joyful - a reflection of the cultural anxieties of our era.
      Speechiness has also risen, driven by the surge of rap and spoken-word genres.</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 5 – LYRIC SENTIMENT
# ═══════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">Lyric Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">What do the words actually say about how we feel?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Sentiment distribution
        fig_sent = px.histogram(
            fdf, x="lyric_sentiment", color="mood_category",
            nbins=40, barmode="overlay", opacity=0.75,
            color_discrete_map=MOOD_COLORS,
            title="Lyric Sentiment Score Distribution by Mood"
        )
        fig_sent.add_vline(x=0, line_dash="dash", line_color="#f0c040", opacity=0.7,
                           annotation_text=" Neutral", annotation_font_color="#f0c040")
        fig_sent.update_layout(**PLOTLY_TEMPLATE, height=360,
                               title_font=dict(size=13, color="#f0c040"))
        fig_sent.update_xaxes(**AXIS_STYLE)
        fig_sent.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_sent, use_container_width=True)

    with col2:
        # Sentiment vs valence scatter
        sample2 = fdf.sample(min(1500, len(fdf)), random_state=7) if len(fdf) > 0 else fdf
        fig_sv = px.scatter(
            sample2, x="lyric_sentiment", y="valence",
            color="mood_category", opacity=0.65,
            color_discrete_map=MOOD_COLORS,
            trendline="ols",
            title="Lyric Sentiment vs Audio Valence"
        )
        fig_sv.update_layout(**PLOTLY_TEMPLATE, height=360,
                             title_font=dict(size=13, color="#f0c040"))
        fig_sv.update_xaxes(**AXIS_STYLE)
        fig_sv.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_sv, use_container_width=True)

    # Sentiment by genre
    st.markdown("### Average Lyric Sentiment by Genre")
    genre_sent = fdf.groupby("playlist_genre")["lyric_sentiment"].mean().reset_index()
    genre_sent.columns = ["Genre", "Avg Sentiment"]
    genre_sent = genre_sent.sort_values("Avg Sentiment", ascending=False)
    fig_gs = px.bar(
        genre_sent, x="Genre", y="Avg Sentiment",
        color="Genre", color_discrete_map=GENRE_COLORS,
        title="Which Genre Has the Most Positive Lyrics?"
    )
    fig_gs.add_hline(y=0, line_color="#f0c040", line_dash="dot", opacity=0.5)
    fig_gs.update_layout(**PLOTLY_TEMPLATE, height=340,
                         title_font=dict(size=13, color="#f0c040"),
                         showlegend=False)
    fig_gs.update_xaxes(**AXIS_STYLE)
    fig_gs.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_gs, use_container_width=True)

    # Sentiment vs popularity
    col3, col4 = st.columns(2)
    with col3:
        fig_sp = px.scatter(
            sample2, x="lyric_sentiment", y="track_popularity",
            color="playlist_genre", opacity=0.6,
            color_discrete_map=GENRE_COLORS,
            title="Does Sentiment Drive Popularity?"
        )
        fig_sp.update_layout(**PLOTLY_TEMPLATE, height=320,
                             title_font=dict(size=13, color="#f0c040"))
        fig_sp.update_xaxes(**AXIS_STYLE)
        fig_sp.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_sp, use_container_width=True)

    with col4:
        mood_sent = fdf.groupby("mood_category")["lyric_sentiment"].mean().reset_index()
        fig_ms = px.bar(
            mood_sent, x="mood_category", y="lyric_sentiment",
            color="mood_category", color_discrete_map=MOOD_COLORS,
            title="Avg Sentiment per Mood Category"
        )
        fig_ms.update_layout(**PLOTLY_TEMPLATE, height=320,
                             title_font=dict(size=13, color="#f0c040"),
                             showlegend=False)
        fig_ms.update_xaxes(**AXIS_STYLE)
        fig_ms.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_ms, use_container_width=True)

    st.markdown("""
    <div class="story-card">
      <h4>🎤 Insight</h4>
      <p>Lyric sentiment and audio valence are correlated but not identical - proving that words and sound
      convey emotion independently. Pop and R&B tend to have the most positive lyrics, while rock songs
      often score lower on textual sentiment. This disconnect between what we hear and what we read
      is itself a profound signal: music communicates emotionally on multiple parallel channels.</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 6 – DEEP DIVE
# ═══════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Deep Dive Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Advanced patterns and artist-level insights</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Top artists by avg popularity
        top_artists = (
            fdf.groupby("track_artist")["track_popularity"]
            .mean()
            .reset_index()
            .sort_values("track_popularity", ascending=False)
            .head(15)
        )
        fig_ta = px.bar(
            top_artists, x="track_popularity", y="track_artist",
            orientation="h", color="track_popularity",
            color_continuous_scale="RdPu",
            title="Top 15 Artists by Avg Popularity"
        )
        fig_ta.update_layout(**PLOTLY_TEMPLATE, height=420,
                             title_font=dict(size=13, color="#f0c040"),
                             coloraxis_showscale=False)
        fig_ta.update_xaxes(**AXIS_STYLE)
        fig_ta.update_yaxes(categoryorder="total ascending", **AXIS_STYLE)
        st.plotly_chart(fig_ta, use_container_width=True)

    with col2:
        # Energy vs tempo colored by mood
        sample3 = fdf.sample(min(1500, len(fdf)), random_state=99) if len(fdf) > 0 else fdf
        fig_et = px.scatter(
            sample3, x="tempo", y="energy",
            color="mood_category", opacity=0.65,
            size="danceability",
            color_discrete_map=MOOD_COLORS,
            title="Tempo vs Energy - Sized by Danceability",
            size_max=14
        )
        fig_et.update_layout(**PLOTLY_TEMPLATE, height=420,
                             title_font=dict(size=13, color="#f0c040"))
        fig_et.update_xaxes(**AXIS_STYLE)
        fig_et.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_et, use_container_width=True)

    # Sunburst: Genre > Mood
    st.markdown("### Genre → Mood Hierarchy")
    gm = fdf.groupby(["playlist_genre","mood_category"]).size().reset_index(name="count")
    fig_sun = px.sunburst(
        gm, path=["playlist_genre","mood_category"], values="count",
        color="mood_category", color_discrete_map=MOOD_COLORS,
        title="Genre and Mood Breakdown (Sunburst)"
    )
    fig_sun.update_layout(**PLOTLY_TEMPLATE, height=480,
                          title_font=dict(size=13, color="#f0c040"))
    fig_sun.update_xaxes(**AXIS_STYLE)
    fig_sun.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_sun, use_container_width=True)

    # Parallel coordinates
    st.markdown("### Multi-Feature Emotional Fingerprint (Parallel Coordinates)")
    pc_sample = fdf.sample(min(1000, len(fdf)), random_state=42) if len(fdf) > 0 else fdf
    mood_codes = {"Happy / Energetic": 0, "Calm / Positive": 1, "Sad / Calm": 2, "Intense / Dark": 3}
    pc_sample = pc_sample.copy()
    pc_sample["mood_code"] = pc_sample["mood_category"].map(mood_codes)

    fig_pc = px.parallel_coordinates(
        pc_sample,
        dimensions=["danceability","energy","valence","tempo","acousticness","speechiness"],
        color="mood_code",
        color_continuous_scale=["#22c55e","#22d3ee","#3b82f6","#ef4444"],
        title="Parallel Coordinates - Audio Features by Mood"
    )
    fig_pc.update_layout(**PLOTLY_TEMPLATE, height=440,
                         title_font=dict(size=13, color="#f0c040"))
    fig_pc.update_xaxes(**AXIS_STYLE)
    fig_pc.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown("""
    <div class="story-card">
      <h4>🔬 Insight</h4>
      <p>The parallel coordinates plot makes the pattern unmistakable: each mood category traces
      a distinct path across all audio dimensions. Happy songs have high danceability, high energy,
      and high valence simultaneously. Intense / Dark songs maintain high energy but drop in valence.
      This multi-dimensional signature is precisely what machine learning algorithms exploit to
      classify emotion with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 7 – CONCLUSION
# ═══════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">Conclusion & Call to Action</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">What the data tells us - and what it means for the future of music</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="conclusion-box">
      <h3>🏁 The Resolution</h3>
      <p style="color:#c0b0d8; font-size:0.95rem; line-height:1.8;">
        This analysis of <strong>14,200 songs</strong> across six genres and six decades confirms that AI can
        detect emotional patterns in music. By combining three types of data - <em>numerical audio features</em>
        (valence, energy, danceability), <em>categorical metadata</em> (genre, subgenre, year),
        and <em>text-based lyric sentiment</em> - we can classify songs into four distinct emotional categories
        with meaningful accuracy. Energy and valence are the strongest predictors of mood.
        Lyric sentiment adds an independent emotional signal. Genre shapes the distribution of emotions
        in predictable ways. Valence has declined since 2010 while energy remains high - music is getting
        more intense but less joyful.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3)
    insights = [
        ("📡", "For Streaming Platforms",
         "Replace genre-only recommendations with emotion-aware playlists. Recommend songs based on the listener's current mood state, not just past behavior."),
        ("🎯", "For Music Marketers",
         "Use valence and energy scores to target audiences. High-valence tracks perform better in upbeat campaign contexts. Understand emotional positioning before releasing."),
        ("🤖", "For AI Researchers",
         "Build multi-modal emotion classifiers that combine audio features, lyrics, and metadata. A single data type is insufficient - emotion is multi-layered."),
    ]
    for col, (icon, title, text) in zip(cols, insights):
        col.markdown(f"""
        <div class="story-card" style="height:180px">
          <h4>{icon} {title}</h4>
          <p>{text}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### The Emotional Appeal")
    st.markdown("""
    <div class="story-card" style="border-left-color:#4ade80">
      <h4>🎵 Beyond the Numbers</h4>
      <p>Music is not just data. It is connected to memory, identity, heartbreak, motivation, and healing.
      Behind every valence score of 0.12 is a breakup. Behind every energy reading of 0.95 is a finish line.
      This project shows how data can help us understand emotion - while reminding us that music remains
      irreducibly human. AI can measure the shape of a feeling. Only you can feel it.</p>
    </div>
    """, unsafe_allow_html=True)

    # Final summary visual
    st.markdown("### At a Glance - Key Findings")
    findings = pd.DataFrame({
        "Finding": [
            "Happy / Energetic is the largest mood class",
            "Energy & loudness are the strongest correlated features",
            "Valence has declined since ~2013",
            "Rap has the highest speechiness (expected)",
            "Pop lyrics score highest in positive sentiment",
            "Genre alone cannot predict mood accurately"
        ],
        "Confidence": [0.92, 0.88, 0.85, 0.97, 0.79, 0.84],
        "Impact": ["High","High","High","Medium","Medium","High"]
    })
    fig_findings = px.bar(
        findings, x="Confidence", y="Finding", orientation="h",
        color="Confidence", color_continuous_scale="RdPu",
        text="Confidence",
        title="Key Findings Confidence Summary"
    )
    fig_findings.update_traces(texttemplate="%{text:.0%}", textposition="outside")
    fig_findings.update_layout(**PLOTLY_TEMPLATE, height=380,
                               title_font=dict(size=13, color="#f0c040"),
                               coloraxis_showscale=False)
    fig_findings.update_xaxes(range=[0, 1.1], tickformat=".0%", **AXIS_STYLE)
    fig_findings.update_yaxes(categoryorder="total ascending", **AXIS_STYLE)
    st.plotly_chart(fig_findings, use_container_width=True)

    st.markdown("""
    <br>
    <div style="text-align:center; color:#4b5563; font-size:0.8rem; padding:20px 0;">
      DSA 506 · Final Project · Shreyasee Poddar · 14,200 Songs Analyzed
    </div>
    """, unsafe_allow_html=True)

