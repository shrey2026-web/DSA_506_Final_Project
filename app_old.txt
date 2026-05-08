import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Can AI Understand Human Emotion Through Music?",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("tableau_music_emotion_dataset.csv")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df

df = load_data()

st.title("Can AI Understand Human Emotion Through Music?")
st.markdown(
    """
    ### Research Question  
    Can song data, including lyrics, audio features, and artist characteristics, help AI identify emotional patterns in modern music?
    """
)

st.markdown("---")

# Sidebar filters
st.sidebar.header("Dashboard Filters")

genres = sorted(df["playlist_genre"].dropna().unique())
selected_genres = st.sidebar.multiselect(
    "Select Genre(s)",
    genres,
    default=genres
)

moods = sorted(df["mood_category"].dropna().unique())
selected_moods = st.sidebar.multiselect(
    "Select Mood Category",
    moods,
    default=moods
)

min_year = int(df["year"].min())
max_year = int(df["year"].max())

year_range = st.sidebar.slider(
    "Select Year Range",
    min_year,
    max_year,
    (2000, 2020)
)

filtered_df = df[
    (df["playlist_genre"].isin(selected_genres)) &
    (df["mood_category"].isin(selected_moods)) &
    (df["year"].between(year_range[0], year_range[1]))
].copy()

# Hook
st.header("The Hook")
st.markdown(
    """
    Music is emotional, personal, and powerful. People use music when they are happy,
    heartbroken, stressed, motivated, or nostalgic. But can AI understand those emotions through data?
    """
)

# KPI cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("Total Songs", f"{filtered_df['track_name'].nunique():,}")
kpi2.metric("Avg Energy", f"{filtered_df['energy'].mean():.2f}")
kpi3.metric("Avg Valence", f"{filtered_df['valence'].mean():.2f}")
kpi4.metric("Avg Popularity", f"{filtered_df['track_popularity'].mean():.1f}")
kpi5.metric("Avg Lyric Sentiment", f"{filtered_df['lyric_sentiment'].mean():.2f}")

st.markdown("---")

# Context
st.header("The Context")
st.markdown(
    """
    This project uses numerical audio features, categorical genre and artist information,
    and text-based lyric sentiment to study emotional patterns in modern music.
    """
)

# Emotional Music Map
st.subheader("Emotional Music Map: Energy vs Valence")

sample_df = filtered_df.sample(
    min(5000, len(filtered_df)),
    random_state=42
) if len(filtered_df) > 0 else filtered_df

fig_map = px.scatter(
    sample_df,
    x="valence",
    y="energy",
    color="mood_category",
    size="track_popularity",
    hover_data=["track_name", "track_artist", "playlist_genre", "year"],
    title="Songs Cluster Into Emotional Zones",
    opacity=0.65,
    range_x=[0, 1],
    range_y=[0, 1]
)

fig_map.update_layout(height=600)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown(
    """
    **Insight:** Energy and valence help translate songs into emotional zones.
    Songs with high energy and high valence appear happier and more energetic,
    while high-energy songs with lower valence appear more intense or darker.
    """
)

st.markdown("---")

# Conflict
st.header("The Conflict")
st.markdown(
    """
    Emotion is subjective. A song can sound energetic but still feel emotionally dark.
    Another song can have sad lyrics but remain popular and danceable.
    This creates a challenge for AI: it can detect emotional signals, but human interpretation is still needed.
    """
)

# Two-column visual section
left, right = st.columns(2)

with left:
    st.subheader("Mood Distribution by Genre")

    mood_genre = (
        filtered_df.groupby(["playlist_genre", "mood_category"])
        .size()
        .reset_index(name="Number of Songs")
    )

    fig_mood = px.bar(
        mood_genre,
        x="playlist_genre",
        y="Number of Songs",
        color="mood_category",
        title="How Mood Categories Differ by Genre"
    )

    fig_mood.update_layout(height=500)
    st.plotly_chart(fig_mood, use_container_width=True)

    st.markdown(
        """
        **Insight:** Genres do not only describe sound. They also carry emotional patterns.
        """
    )

with right:
    st.subheader("Average Audio Features by Genre")

    genre_profile = (
        filtered_df.groupby("playlist_genre")[["energy", "valence", "danceability"]]
        .mean()
        .reset_index()
    )

    genre_profile_long = genre_profile.melt(
        id_vars="playlist_genre",
        value_vars=["energy", "valence", "danceability"],
        var_name="Audio Feature",
        value_name="Average Score"
    )

    fig_profile = px.bar(
        genre_profile_long,
        x="playlist_genre",
        y="Average Score",
        color="Audio Feature",
        barmode="group",
        title="Average Energy, Valence, and Danceability by Genre",
        range_y=[0, 1]
    )

    fig_profile.update_layout(height=500)
    st.plotly_chart(fig_profile, use_container_width=True)

    st.markdown(
        """
        **Insight:** Each genre has a measurable emotional and musical profile.
        """
    )

st.markdown("---")

# Journey
st.header("The Journey")
st.markdown(
    """
    The analysis began by cleaning the dataset and filtering English songs.
    Then numerical audio features were used to measure energy, positivity, rhythm, and popularity.
    Genre and artist were used as categorical variables, while lyric sentiment added a text-based emotional layer.
    """
)

# Sentiment and time trend
left2, right2 = st.columns(2)

with left2:
    st.subheader("Lyric Sentiment by Genre")

    sentiment_genre = (
        filtered_df.groupby("playlist_genre")["lyric_sentiment"]
        .mean()
        .reset_index()
    )

    fig_sentiment = px.bar(
        sentiment_genre,
        x="playlist_genre",
        y="lyric_sentiment",
        title="Average Lyric Sentiment by Genre"
    )

    fig_sentiment.update_layout(height=500)
    st.plotly_chart(fig_sentiment, use_container_width=True)

    st.markdown(
        """
        **Insight:** Lyrics add human context, but audio features are also needed
        to understand emotion more completely.
        """
    )

with right2:
    st.subheader("Emotional Trends Over Time")

    yearly = (
        filtered_df.groupby("year")[["energy", "valence", "danceability"]]
        .mean()
        .reset_index()
    )

    yearly_long = yearly.melt(
        id_vars="year",
        value_vars=["energy", "valence", "danceability"],
        var_name="Audio Feature",
        value_name="Average Score"
    )

    fig_trend = px.line(
        yearly_long,
        x="year",
        y="Average Score",
        color="Audio Feature",
        markers=True,
        title="How Music Emotion Changes Over Time",
        range_y=[0, 1]
    )

    fig_trend.update_layout(height=500)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown(
        """
        **Insight:** Music remains energetic and danceable over time,
        while emotional positivity changes across years.
        """
    )

st.markdown("---")

# Animation section
st.header("Animation: Emotional Patterns Over Time")
st.markdown(
    """
    This animation shows how songs appear in emotional space across release years.
    It has a clear purpose: showing whether music emotion changes over time.
    """
)

anim_df = filtered_df[
    (filtered_df["year"] >= year_range[0]) &
    (filtered_df["year"] <= year_range[1])
].copy()

if len(anim_df) > 4000:
    anim_df = anim_df.sample(4000, random_state=42)

fig_anim = px.scatter(
    anim_df,
    x="valence",
    y="energy",
    animation_frame="year",
    color="playlist_genre",
    size="track_popularity",
    hover_name="track_name",
    hover_data=["track_artist", "mood_category"],
    range_x=[0, 1],
    range_y=[0, 1],
    title="Animated Emotional Music Map Over Time",
    opacity=0.70
)

fig_anim.update_layout(height=650)
st.plotly_chart(fig_anim, use_container_width=True)

# Animated time-series
st.subheader("Animated Time-Series: Emotional Audio Features")

yearly_anim = (
    filtered_df.groupby("year")[["energy", "valence", "danceability"]]
    .mean()
    .reset_index()
)

yearly_anim_long = yearly_anim.melt(
    id_vars="year",
    value_vars=["energy", "valence", "danceability"],
    var_name="Audio Feature",
    value_name="Average Score"
)

frames = []
for frame_year in sorted(yearly_anim_long["year"].unique()):
    temp = yearly_anim_long[yearly_anim_long["year"] <= frame_year].copy()
    temp["frame_year"] = frame_year
    frames.append(temp)

if frames:
    yearly_anim_cumulative = pd.concat(frames)

    fig_time_anim = px.line(
        yearly_anim_cumulative,
        x="year",
        y="Average Score",
        color="Audio Feature",
        animation_frame="frame_year",
        markers=True,
        range_y=[0, 1],
        title="Animated Time-Series of Emotional Audio Features"
    )

    fig_time_anim.update_layout(height=550)
    st.plotly_chart(fig_time_anim, use_container_width=True)

st.markdown("---")

# Resolution
st.header("The Resolution")
st.markdown(
    """
    The analysis shows that AI can identify emotional patterns in music by combining
    audio features, lyrics, and genre information. Energy and valence create simple
    mood categories, while lyric sentiment adds another emotional layer.
    """
)

# Call to Action
st.header("The Call to Action")
st.markdown(
    """
    Streaming platforms, playlist creators, and music marketers can use emotion-based
    music analysis to improve recommendations. Instead of recommending songs only by genre
    or popularity, platforms can recommend music based on emotional needs such as motivation,
    relaxation, nostalgia, or mood recovery.
    """
)

# Emotional Appeal
st.header("The Emotional Appeal")
st.markdown(
    """
    Music is not just data. It is connected to memory, identity, heartbreak, motivation,
    and healing. Behind every song is a human feeling. This project shows how data can help
    us understand emotion, while reminding us that music remains deeply personal.
    """
)

st.markdown("---")

st.success(
    "Conclusion: AI can detect emotional signals in music, but human interpretation is still needed to understand why music matters."
)
