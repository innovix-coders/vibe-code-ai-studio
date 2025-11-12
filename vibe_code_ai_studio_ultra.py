# ============================================================
# 💎 VIBE-CODE AI STUDIO ULTRA — by Team Innovex Coders
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Vibe-Code AI Studio Ultra", page_icon="💎", layout="wide")

# ------------------------------------------------------------
# BUILT-IN DATASETS
# ------------------------------------------------------------
DATASETS = {
    "Sales Data": pd.DataFrame({
        "Region": ["North", "South", "East", "West", "Central"],
        "Sales": [120, 150, 100, 180, 140],
        "Profit": [30, 40, 25, 60, 35],
        "Year": [2020, 2021, 2022, 2023, 2024]
    }),
    "Student Performance": pd.DataFrame({
        "Student": ["A", "B", "C", "D", "E", "F"],
        "Math": [85, 90, 75, 60, 95, 70],
        "Science": [88, 76, 85, 60, 90, 80],
        "English": [78, 89, 70, 68, 92, 77]
    }),
    "Climate Data": pd.DataFrame({
        "Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "Temperature": [20,22,25,28,30,32,33,31,29,26,23,21],
        "Rainfall": [100,80,60,40,20,10,15,30,50,70,90,110]
    }),
    "Website Analytics": pd.DataFrame({
        "Month": ["Jan","Feb","Mar","Apr","May","Jun"],
        "Visitors": [1200,1500,1700,1600,2000,2500],
        "Conversions": [80,90,100,95,120,140],
        "Bounce Rate": [60,55,52,50,48,45]
    })
}

# ------------------------------------------------------------
# INTENT & THEMES
# ------------------------------------------------------------
INTENT_KEYWORDS = {
    "Comparison": ["compare","versus","across","rank"],
    "Trend": ["trend","increase","decrease","growth","year","time"],
    "Composition": ["share","portion","contribution","part"],
    "Distribution": ["distribution","spread","range","variance"],
    "Relationship": ["relationship","correlation","link","impact"]
}
THEMES = {"Dark": "plotly_dark", "Light": "plotly_white", "Minimal": "simple_white"}

# ------------------------------------------------------------
# INTELLIGENCE FUNCTIONS
# ------------------------------------------------------------
def detect_intent(goal: str) -> str:
    goal = goal.lower()
    for intent, keys in INTENT_KEYWORDS.items():
        if any(k in goal for k in keys):
            return intent
    return "Comparison"

def recommend_chart(intent: str):
    charts = {"Comparison": "Bar","Trend": "Line","Composition": "Pie",
              "Distribution": "Histogram","Relationship": "Scatter"}
    return charts.get(intent, "Bar")

# -------- Deep AI Summary --------
def ai_summary(df, x, y, intent):
    """🧠 Deep AI Analytical Summary"""
    try:
        lines = []
        n_rows, n_cols = df.shape

        # --- General information ---
        lines.append(f"### 🧩 Data Summary")
        lines.append(f"- Records analyzed: **{n_rows}**, Columns: **{n_cols}**")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if num_cols:
            lines.append(f"- Numeric fields: {', '.join(num_cols)}")
        if cat_cols:
            lines.append(f"- Categorical fields: {', '.join(cat_cols)}")

        # --- Statistics for y ---
        desc = df[y].describe()
        lines.append("\n### 📊 Statistical Overview")
        lines.append(f"- Mean: {desc['mean']:.2f}, Min: {desc['min']:.2f}, Max: {desc['max']:.2f}, Std: {desc['std']:.2f}")

        # --- Extremes ---
        top = df.loc[df[y].idxmax(), x]
        low = df.loc[df[y].idxmin(), x]
        gap = df[y].max() - df[y].min()
        lines.append("\n### 🏆 Performance Highlights")
        lines.append(f"- Highest **{y}** in **{top}** ({df[y].max():.2f})")
        lines.append(f"- Lowest **{y}** in **{low}** ({df[y].min():.2f})")
        lines.append(f"- Difference between max and min: {gap:.2f}")

        # --- Trend analysis ---
        if "Year" in df.columns:
            yearly = df.groupby("Year")[y].mean().reset_index()
            change = round(((yearly[y].iloc[-1]-yearly[y].iloc[0]) / yearly[y].iloc[0])*100,1)
            direction = "increased 📈" if change>0 else "decreased 📉"
            lines.append("\n### 📈 Trend Analysis")
            lines.append(f"- Over time, **{y}** has {direction} by **{abs(change)}%**.")

        # --- Variability ---
        ratio = df[y].std() / (df[y].mean() if df[y].mean()!=0 else 1)
        if ratio > 0.5:
            lines.append("\n### ⚠️ Variability Insight")
            lines.append(f"- High spread detected (Std/Mean ratio {ratio:.2f}); consider box or histogram views.")
        else:
            lines.append("\n### ✅ Stability Insight")
            lines.append(f"- Moderate variation (Std/Mean ratio {ratio:.2f}); data appears consistent.")

        # --- Correlation ---
        num_df = df.select_dtypes(include=np.number)
        if len(num_df.columns) > 1:
            corr = num_df.corr()
            top_corr = corr.unstack().sort_values(ascending=False)
            top_corr = top_corr[top_corr < 1].head(3)
            lines.append("\n### 🔗 Correlation Highlights")
            for (a,b), val in top_corr.items():
                lines.append(f"- **{a}** ↔ **{b}** : {val:.2f}")

        # --- Outlier detection ---
        q1,q3 = df[y].quantile([0.25,0.75])
        iqr = q3 - q1
        outliers = df[(df[y] < q1-1.5*iqr) | (df[y] > q3+1.5*iqr)]
        if not outliers.empty:
            lines.append("\n### 🚨 Outlier Detection")
            lines.append(f"- {len(outliers)} potential outliers found in **{y}**.")

        # --- AI interpretation ---
        lines.append("\n### 🧠 AI Interpretation")
        if intent=="Comparison":
            lines.append(f"The chart highlights category-wise differences in **{y}**; **{top}** leads significantly.")
        elif intent=="Trend":
            lines.append(f"The time-based pattern shows how **{y}** evolves — possibly driven by seasonality or policy shifts.")
        elif intent=="Composition":
            lines.append(f"The distribution of **{y}** reveals disproportionate contributions across {x}.")
        elif intent=="Distribution":
            lines.append(f"Spread of **{y}** values suggests {'skewed' if ratio>0.5 else 'balanced'} distribution.")
        elif intent=="Relationship":
            lines.append(f"Correlations indicate dependencies among variables, useful for predictive insights.")
        else:
            lines.append("The data shows both stable and varying patterns depending on category context.")

        lines.append("\n---\n**🔍 Conclusion:** This visualization exposes major performance gaps, variability, and key drivers. Consider deeper analytics or forecasting for strategic decisions.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error generating report: {e}"

# -------- Chart creation --------
def create_chart(df, chart_type, x, y, theme):
    title = f"{chart_type} of {y} vs {x}"
    if chart_type=="Bar":
        return px.bar(df,x=x,y=y,color=x,title=title,template=theme)
    elif chart_type=="Line":
        return px.line(df,x=x,y=y,markers=True,title=title,template=theme)
    elif chart_type=="Pie":
        return px.pie(df,names=x,values=y,hole=0.3,title=title,template=theme)
    elif chart_type=="Area":
        return px.area(df,x=x,y=y,title=title,template=theme)
    elif chart_type=="Scatter":
        return px.scatter(df,x=x,y=y,size=y,color=x,title=title,template=theme)
    elif chart_type=="Histogram":
        return px.histogram(df,x=y,nbins=15,title=title,template=theme)
    return px.bar(df,x=x,y=y,title="Chart error",template=theme)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

examples = [
    "Compare regional sales and profits",
    "Show yearly sales growth trend",
    "Analyze composition of revenue by region",
    "Understand distribution of profit margins",
    "Check relationship between expenses and sales"
]
goal = st.sidebar.text_area("🧠 Your Data Goal", placeholder=np.random.choice(examples))
dataset_name = st.sidebar.selectbox("📂 Choose Dataset", list(DATASETS.keys()))
theme_choice = st.sidebar.radio("🎨 Theme", list(THEMES.keys()), horizontal=True)
chart_choice = st.sidebar.selectbox("📈 Chart Type", ["Auto","Bar","Line","Pie","Area","Scatter","Histogram"])
uploaded_file = st.sidebar.file_uploader("⬆️ Upload CSV/Excel", type=["csv","xlsx"])
st.sidebar.caption("💡 Made with ❤️ by Team Innovex Coders")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
else:
    df = DATASETS[dataset_name]

# ------------------------------------------------------------
# DATA OVERVIEW
# ------------------------------------------------------------
st.markdown("## 📊 Data Overview")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.dataframe(df.head())
st.markdown("### 📈 Summary Statistics")
st.dataframe(df.describe())

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

# ------------------------------------------------------------
# INTENT + VISUALIZATION
# ------------------------------------------------------------
intent = detect_intent(goal)
auto_chart = recommend_chart(intent)
final_chart = chart_choice if chart_choice!="Auto" else auto_chart
theme = THEMES[theme_choice]

x = st.selectbox("X-Axis", cat_cols + num_cols)
y = st.selectbox("Y-Axis", num_cols)

st.markdown(f"## 🎯 Visualization — {final_chart} Chart ({intent} Intent)")
fig = create_chart(df, final_chart, x, y, theme)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# AI INSIGHTS
# ------------------------------------------------------------
st.markdown("## 🧠 AI Analytical Report")
st.markdown(ai_summary(df, x, y, intent))

# ------------------------------------------------------------
# HEATMAP + RECOMMENDATIONS
# ------------------------------------------------------------
if len(num_cols) > 1:
    st.markdown("## 🔥 Correlation Heatmap")
    corr_fig = px.imshow(df[num_cols].corr(), text_auto=True,
                         color_continuous_scale="RdBu_r",
                         title="Numeric Correlations")
    st.plotly_chart(corr_fig, use_container_width=True)

st.markdown("## 💡 Smart Recommendations")
tips = {
    "Trend":"Try Line/Area charts for time trends.",
    "Comparison":"Bar charts best show categorical differences.",
    "Composition":"Pie or stacked bar charts show parts of whole.",
    "Distribution":"Use histograms or box plots for spread.",
    "Relationship":"Scatter plots reveal variable interaction."
}
st.success(tips.get(intent,"Choose clear charts and consistent color schemes."))

st.markdown("---")
st.markdown("<center>✨ Made with ❤️ by <b>Team Innovex Coders</b> | Powered by Streamlit + Plotly</center>", unsafe_allow_html=True)
