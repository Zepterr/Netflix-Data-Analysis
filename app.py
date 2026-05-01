import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');

html, body, .stApp { background-color: #141414 !important; }

section[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    border-right: 2px solid #E50914;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }

/* Sidebar collapse button */
[data-testid="collapsedControl"] {
    background-color: #E50914 !important;
    color: white !important;
}
button[kind="header"] { background-color: #E50914 !important; }

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #1f1f1f !important;
    border: 1px solid #E50914 !important;
    color: #ffffff !important;
}
div[data-baseweb="select"] span { color: #ffffff !important; }
div[data-baseweb="popover"] div { 
    background-color: #1f1f1f !important; 
    color: #ffffff !important; 
}
li[role="option"] { 
    background-color: #1f1f1f !important; 
    color: #ffffff !important; 
}
li[role="option"]:hover { background-color: #E50914 !important; }
li[aria-selected="true"] { background-color: #8b0000 !important; }

/* Headings */
h1, h2, h3, h4 {
    font-family: 'Bebas Neue', sans-serif !important;
    color: #E50914 !important;
    letter-spacing: 3px;
}
p, div, span, label { 
    color: #ffffff !important; 
    font-family: 'Inter', sans-serif !important; 
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a1a, #2d0808) !important;
    border: 1px solid #E50914 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
[data-testid="stMetricLabel"] p { color: #aaaaaa !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #E50914 !important; font-size: 30px !important; font-weight: 700 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0d0d0d !important;
    border-bottom: 2px solid #E50914 !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #888 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 12px 18px !important;
    border-radius: 4px 4px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background-color: #E50914 !important;
}

/* Text input */
.stTextInput input {
    background-color: #1a1a1a !important;
    border: 2px solid #E50914 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-size: 15px !important;
}
.stTextInput input::placeholder { color: #555 !important; }

/* Slider */
[data-testid="stSlider"] * { color: #ffffff !important; }
.stSlider [data-baseweb="slider"] div { background: #E50914 !important; }

/* Divider */
hr { border-color: #E50914 !important; opacity: 0.2; margin: 16px 0; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #141414; }
::-webkit-scrollbar-thumb { background: #E50914; border-radius: 3px; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #333 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
    df = df.dropna(subset=['duration', 'date_added'])
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip())
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['month_name'] = df['date_added'].dt.strftime('%B')
    df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(int)
    df['release_year'] = df['release_year'].astype(int)
    return df

df = load_data()

# ── Recommender ───────────────────────────────────────────
@st.cache_data
def build_recommender(df):
    features = (
        df['listed_in'].fillna('') + ' ' +
        df['director'].fillna('') + ' ' +
        df['cast'].fillna('') + ' ' +
        df['description'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = tfidf.fit_transform(features)
    sim = cosine_similarity(matrix, matrix)
    idx = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
    return sim, idx

cosine_sim, indices = build_recommender(df)

def get_recommendations(title, n=5):
    t = title.lower().strip()
    if t not in indices:
        return None, None
    i = indices[t]
    scores = sorted(list(enumerate(cosine_sim[i])),
                    key=lambda x: x[1], reverse=True)[1:n+1]
    input_row = df.iloc[i]
    results = []
    for mi, score in scores:
        row = df.iloc[mi]
        results.append({
            'Title': row['title'],
            'Type': row['type'],
            'Genre': row['listed_in'],
            'Rating': row['rating'],
            'Year': int(row['release_year']),
            'Match Score': f"{score:.2%}"
        })
    return pd.DataFrame(results), input_row

# ── Shared plot layout (no yaxis/xaxis here to avoid conflicts) ──
def base_layout(title=''):
    return dict(
        paper_bgcolor='#141414',
        plot_bgcolor='#1a1a1a',
        font=dict(color='white', family='Inter'),
        title=dict(text=title, font=dict(color='#E50914', size=20)),
        margin=dict(t=60, b=40, l=40, r=40),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#333',
                    borderwidth=1, font=dict(color='white')),
    )

def apply_axes(fig, xgrid=True, ygrid=True):
    fig.update_xaxes(gridcolor='#2a2a2a', showgrid=xgrid, zeroline=False, color='white')
    fig.update_yaxes(gridcolor='#2a2a2a', showgrid=ygrid, zeroline=False, color='white')
    return fig

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#E50914; font-size:30px; letter-spacing:3px;'>🎬 NETFLIX</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa; font-size:12px; margin-top:-12px;'>Analytics Dashboard · Spring 2026</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#E50914; font-size:10px; font-weight:700; letter-spacing:2px;'>▶ FILTERS</p>", unsafe_allow_html=True)

    content_type = st.selectbox("Content Type", ["All", "Movie", "TV Show"])
    year_range = st.slider(
        "Year Added",
        min_value=2015,
        max_value=2021,
        value=(2015, 2021)
    )
    all_ratings = ["All"] + sorted(df['rating'].dropna().unique().tolist())
    selected_rating = st.selectbox("Content Rating", all_ratings)

    st.markdown("---")
    st.markdown("<p style='color:#E50914; font-size:10px; font-weight:700; letter-spacing:2px;'>▶ DATASET INFO</p>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#aaa; font-size:11px; line-height:2;'>
    📊 Source: Kaggle — Netflix Titles<br>
    📁 8,794 titles · 12 features<br>
    🌍 127 countries represented<br>
    📅 Data up to 2021
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#E50914; font-size:10px; font-weight:700; letter-spacing:2px;'>▶ TEAM</p>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#aaa; font-size:11px; line-height:2;'>
    🎓 Intro to Data Science<br>
    Syed Muhammad Zohaib<br>
    Laraib Fatima<br>
    Shahmir Khan<br>
    Syed Jawad
    </p>
    """, unsafe_allow_html=True)

# ── Apply filters ─────────────────────────────────────────
fdf = df.copy()
if content_type != "All":
    fdf = fdf[fdf['type'] == content_type]
fdf = fdf[(fdf['year_added'] >= year_range[0]) & (fdf['year_added'] <= year_range[1])]
if selected_rating != "All":
    fdf = fdf[fdf['rating'] == selected_rating]

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px 0 8px 0;'>
    <h1 style='font-size:54px; letter-spacing:6px; margin:0; color:#E50914;'>
        🎬 NETFLIX ANALYTICS DASHBOARD
    </h1>
    <p style='color:#aaa; font-size:14px; margin-top:6px;'>
        Discovering Patterns & Insights in Streaming Content &nbsp;·&nbsp;
        Introduction to Data Science &nbsp;·&nbsp; Spring 2026
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
movies_f = fdf[fdf['type']=='Movie']
shows_f  = fdf[fdf['type']=='TV Show']
avg_dur  = int(movies_f['duration_int'].mean()) if len(movies_f) > 0 else 0

k1.metric("Total Titles",    f"{len(fdf):,}")
k2.metric("Movies",          f"{len(movies_f):,}")
k3.metric("TV Shows",        f"{len(shows_f):,}")
k4.metric("Countries",       f"{fdf[fdf['country']!='Unknown']['country'].str.split(', ').explode().nunique()}")
k5.metric("Unique Genres",   f"{fdf['listed_in'].str.split(', ').explode().nunique()}")
k6.metric("Avg Movie Length",f"{avg_dur} min")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview",
    "🌍  Geography",
    "🎭  Genres & Ratings",
    "📅  Trends",
    "🤖  Recommender"
])

# ════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        tc = fdf['type'].value_counts().reset_index()
        tc.columns = ['Type','Count']
        fig = px.pie(tc, values='Count', names='Type',
                     hole=0.55,
                     color_discrete_sequence=['#E50914','#564d4d'])
        fig.update_traces(textfont_size=14, textfont_color='white',
                          pull=[0.04,0.04],
                          texttemplate='<b>%{label}</b><br>%{percent}')
        fig.update_layout(**base_layout('Content Type Distribution'))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        yearly = fdf.groupby(['year_added','type']).size().reset_index(name='Count')
        fig = px.line(yearly, x='year_added', y='Count', color='type',
                      markers=True,
                      color_discrete_sequence=['#E50914','#ffffff'],
                      labels={'year_added':'Year','Count':'Titles Added','type':'Type'})
        fig.update_traces(line_width=3, marker_size=10)
        fig.update_layout(**base_layout('Content Growth Over Time'))
        apply_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

    rating_order = ['G','PG','PG-13','R','NC-17','TV-Y','TV-Y7',
                    'TV-Y7-FV','TV-G','TV-PG','TV-14','TV-MA']
    rt = fdf.groupby(['rating','type']).size().reset_index(name='Count')
    rt = rt[rt['rating'].isin(rating_order)]
    fig = px.bar(rt, x='rating', y='Count', color='type',
                 barmode='stack',
                 color_discrete_sequence=['#E50914','#564d4d'],
                 category_orders={'rating': rating_order},
                 labels={'rating':'Rating','Count':'Titles','type':'Type'})
    fig.update_layout(**base_layout('Rating Distribution by Content Type'))
    apply_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    if len(movies_f) > 0:
        fig = px.histogram(movies_f, x='duration_int', nbins=40,
                           color_discrete_sequence=['#E50914'],
                           labels={'duration_int':'Duration (minutes)'})
        mean_d   = movies_f['duration_int'].mean()
        median_d = movies_f['duration_int'].median()
        fig.add_vline(x=mean_d, line_dash='dash', line_color='white', line_width=2,
                      annotation_text=f'Mean: {mean_d:.0f} min',
                      annotation_font_color='white',
                      annotation_position='top right')
        fig.add_vline(x=median_d, line_dash='dot', line_color='#ffaa00', line_width=2,
                      annotation_text=f'Median: {median_d:.0f} min',
                      annotation_font_color='#ffaa00',
                      annotation_position='top left')
        fig.update_layout(**base_layout('Movie Duration Distribution'))
        apply_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════
# TAB 2 — GEOGRAPHY
# ════════════════════════════════════
with tab2:
    cs = fdf[fdf['country']!='Unknown']['country'].str.split(', ').explode()
    tc = cs.value_counts().head(15).reset_index()
    tc.columns = ['Country','Count']
    fig = px.bar(tc, x='Count', y='Country', orientation='h',
                 color='Count', color_continuous_scale='Reds', text='Count')
    fig.update_traces(textfont_color='white', textposition='outside')
    fig.update_layout(**base_layout('Top 15 Content-Producing Countries'),
                      coloraxis_showscale=False)
    fig.update_yaxes(categoryorder='total ascending', color='white')
    fig.update_xaxes(gridcolor='#2a2a2a', color='white')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        td = fdf[fdf['director']!='Unknown']['director']\
            .str.split(', ').explode().value_counts().head(10).reset_index()
        td.columns = ['Director','Titles']
        fig = px.bar(td, x='Titles', y='Director', orientation='h',
                     color='Titles', color_continuous_scale='Reds', text='Titles')
        fig.update_traces(textfont_color='white', textposition='outside')
        fig.update_layout(**base_layout('Top 10 Directors'), coloraxis_showscale=False)
        fig.update_yaxes(categoryorder='total ascending', color='white')
        fig.update_xaxes(gridcolor='#2a2a2a', color='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ca = fdf[fdf['cast']!='Unknown']['cast']\
            .str.split(', ').explode().value_counts().head(10).reset_index()
        ca.columns = ['Actor','Appearances']
        fig = px.bar(ca, x='Appearances', y='Actor', orientation='h',
                     color='Appearances', color_continuous_scale='Reds', text='Appearances')
        fig.update_traces(textfont_color='white', textposition='outside')
        fig.update_layout(**base_layout('Top 10 Actors'), coloraxis_showscale=False)
        fig.update_yaxes(categoryorder='total ascending', color='white')
        fig.update_xaxes(gridcolor='#2a2a2a', color='white')
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════
# TAB 3 — GENRES & RATINGS
# ════════════════════════════════════
with tab3:
    gs = fdf['listed_in'].str.split(', ').explode()
    tg = gs.value_counts().head(12).reset_index()
    tg.columns = ['Genre','Count']
    fig = px.bar(tg, x='Count', y='Genre', orientation='h',
                 color='Count', color_continuous_scale='Reds', text='Count')
    fig.update_traces(textfont_color='white', textposition='outside')
    fig.update_layout(**base_layout('Top 12 Genres on Netflix'), coloraxis_showscale=False)
    fig.update_yaxes(categoryorder='total ascending', color='white')
    fig.update_xaxes(gridcolor='#2a2a2a', color='white')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        mg = fdf[fdf['type']=='Movie']['listed_in'].str.split(', ')\
            .explode().value_counts().head(8).reset_index()
        mg.columns = ['Genre','Count']
        fig = px.pie(mg, values='Count', names='Genre',
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_traces(textfont_color='white', textfont_size=11)
        fig.update_layout(**base_layout('Top Movie Genres'))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tg2 = fdf[fdf['type']=='TV Show']['listed_in'].str.split(', ')\
            .explode().value_counts().head(8).reset_index()
        tg2.columns = ['Genre','Count']
        fig = px.pie(tg2, values='Count', names='Genre',
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_traces(textfont_color='white', textfont_size=11)
        fig.update_layout(**base_layout('Top TV Show Genres'))
        st.plotly_chart(fig, use_container_width=True)

    rc = fdf['rating'].value_counts().reset_index()
    rc.columns = ['Rating','Count']
    fig = px.bar(rc, x='Rating', y='Count',
                 color='Count', color_continuous_scale='Reds', text='Count')
    fig.update_traces(textfont_color='white', textposition='outside')
    fig.update_layout(**base_layout('Overall Rating Distribution'), coloraxis_showscale=False)
    fig.update_xaxes(color='white')
    fig.update_yaxes(gridcolor='#2a2a2a', color='white')
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════
# TAB 4 — TRENDS
# ════════════════════════════════════
with tab4:
    month_order = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']
    hmap = fdf[fdf['year_added']>=2015].groupby(
        ['year_added','month_name']).size().reset_index(name='Count')
    hp = hmap.pivot(index='year_added', columns='month_name', values='Count')
    hp = hp.reindex(columns=month_order, fill_value=0)
    fig = px.imshow(hp, color_continuous_scale='Reds',
                    text_auto=True, aspect='auto')
    fig.update_layout(**base_layout('Monthly Content Additions Heatmap (2015–2021)'))
    fig.update_xaxes(color='white')
    fig.update_yaxes(color='white')
    st.plotly_chart(fig, use_container_width=True)

    top6 = ['Dramas','Comedies','Documentaries',
            'Action & Adventure','International Movies','Children & Family Movies']
    ge = fdf[fdf['year_added']>=2015].copy()
    ge = ge.assign(genre=ge['listed_in'].str.split(', ')).explode('genre')
    gy = ge[ge['genre'].isin(top6)]\
        .groupby(['year_added','genre']).size().reset_index(name='Count')
    fig = px.line(gy, x='year_added', y='Count', color='genre',
                  markers=True,
                  color_discrete_sequence=['#E50914','#ff6b6b','#ffffff',
                                           '#ffaa00','#00cfff','#aaffaa'],
                  labels={'year_added':'Year','Count':'Titles','genre':'Genre'})
    fig.update_traces(line_width=2.5, marker_size=8)
    fig.update_layout(**base_layout('Genre Trends Over Time (2015–2021)'))
    apply_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    rd = fdf[fdf['release_year']>=1990]['release_year']\
        .value_counts().sort_index().reset_index()
    rd.columns = ['Release Year','Count']
    fig = px.area(rd, x='Release Year', y='Count',
                  color_discrete_sequence=['#E50914'])
    fig.update_traces(fill='tozeroy', fillcolor='rgba(229,9,20,0.15)',
                      line_color='#E50914')
    fig.update_layout(**base_layout('Content Release Year Distribution (1990–2021)'))
    apply_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════
# TAB 5 — RECOMMENDER
# ════════════════════════════════════
with tab5:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a1a,#2d0808);
                border:1px solid #E50914; border-radius:12px;
                padding:20px; margin-bottom:20px;'>
        <h3 style='color:#E50914; margin:0; font-size:26px; letter-spacing:2px;'>
            🤖 CONTENT-BASED RECOMMENDATION ENGINE
        </h3>
        <p style='color:#aaa; margin:8px 0 0 0; font-size:13px;'>
            TF-IDF Vectorization &amp; Cosine Similarity · 8,794 titles · 15,000 features
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3,1])
    with col1:
        title_input = st.text_input(
            "Enter any Netflix title:",
            placeholder="Try: Inception · Stranger Things · The Irishman · Dark · Money Heist"
        )
    with col2:
        n_recs = st.selectbox("Results", [5, 8, 10])

    if title_input:
        with st.spinner('Finding similar titles...'):
            results, input_row = get_recommendations(title_input, n=n_recs)

        if results is not None:
            i1,i2,i3,i4 = st.columns(4)
            i1.metric("Input Title", input_row['title'][:18] +
                      ('...' if len(input_row['title'])>18 else ''))
            i2.metric("Type",   input_row['type'])
            i3.metric("Rating", input_row['rating'])
            i4.metric("Year",   int(input_row['release_year']))

            st.markdown("#### Recommended Titles")
            st.dataframe(results, use_container_width=True, hide_index=True)

            fig = px.bar(results, x='Match Score', y='Title',
                         orientation='h',
                         color_discrete_sequence=['#E50914'],
                         text='Match Score')
            fig.update_traces(textfont_color='white', textposition='outside')
            fig.update_layout(**base_layout('Similarity Match Scores'))
            fig.update_yaxes(categoryorder='total ascending', color='white')
            fig.update_xaxes(gridcolor='#2a2a2a', color='white')
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"❌ '{title_input}' not found in dataset.")
            matches = [df.loc[indices[t],'title']
                       for t in indices.index if title_input.lower() in t]
            if matches:
                st.info("💡 Did you mean:\n\n" + "\n\n".join([f"• {m}" for m in matches[:6]]))
            else:
                st.warning("Try: Inception · Stranger Things · Money Heist · Dark · The Irishman")

st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#333; font-size:12px;'>
    Netflix Data Analysis · Introduction to Data Science · Spring 2026 ·
    Zohaib · Laraib · Shahmir · Jawad
</p>
""", unsafe_allow_html=True)