# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import difflib
import io
import re
from datetime import datetime
from functools import reduce
from collections import Counter

# =========================================================
# Vibe-Code AI Studio Ultra — POWERFUL AI SUMMARY EDITION
# - **FIXED:** UnboundLocalError by initializing 'outlier_suggestion'.
# - AI Summary now includes: Trend Interpretation, Categorical Dominance, Cross-Correlation.
# - AI Controls and other Extraordinary features retained.
# =========================================================

st.set_page_config(page_title="Vibe-Code AI Studio Ultra — Powerful AI Summary", page_icon="💡", layout="wide")

# -----------------------------
# Built-in datasets (examples)
# -----------------------------
DATASETS = {
    "Student Performance": pd.DataFrame({
        "Student": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "Math": [85, 90, 75, 60, 95, 70, 80, 55],
        "Science": [88, 76, 85, 60, 90, 80, 95, 50],
        "English": [78, 89, 70, 68, 92, 77, 85, 65],
        "Hours Studied": [10, 12, 8, 5, 15, 7, 11, 4]
    }),
    "Sales Data": pd.DataFrame({
        "Region": ["North", "South", "East", "West", "Central"] * 2,
        "Date": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01"][:10]),
        "Sales": [120, 150, 100, 180, 140, 130, 160, 110, 190, 150][:10],
        "Profit": [30, 40, 25, 60, 35, 32, 45, 28, 65, 38][:10],
        "Cost": [90, 110, 75, 120, 105, 98, 115, 82, 125, 112][:10]
    })
}

THEMES = {"Dark": "plotly_dark", "Light": "plotly_white", "Minimal": "simple_white"}
PALETTES = ["plotly", "ggplot2", "viridis", "cividis"]

# -----------------------------
# Helpers: normalization, fuzzy matching, parsing
# -----------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if 'date' in col.lower() or 'year' in col.lower() and df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    return df

def fuzzy_match_token(token: str, choices: list, cutoff=0.6):
    if not token or not choices:
        return None
    matches = difflib.get_close_matches(token, choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def parse_goal(goal: str, name_choices: list, subject_choices: list):
    goal = (goal or "").strip()
    goal_low = goal.lower()

    # intent
    if any(x in goal_low for x in [" vs ", " versus ", " compare ", " against ", " vs. ", " v "]):
        intent = "compare"
    elif any(x in goal_low for x in ["forecast", "predict", "extrapolate", "future"]):
        intent = "forecast"
    elif any(x in goal_low for x in ["trend", "growth", "increase", "decrease", "year", "time", "date"]):
        intent = "trend"
    elif any(x in goal_low for x in ["distribution", "histogram", "spread", "box"]):
        intent = "distribution"
    elif any(x in goal_low for x in ["compose", "composition", "share", "treemap", "sunburst"]):
        intent = "composition"
    elif any(x in goal_low for x in ["scatter", "relationship", "relation", "driver", "factors"]):
        intent = "relationship"
    else:
        intent = "insight"
    
    # subjects detection (exact word boundaries)
    subjects = []
    for s in subject_choices:
        if re.search(rf"\b{re.escape(str(s).lower())}\b", goal_low):
            subjects.append(s)

    # find potential names (tokens that fuzzily match name choices)
    tokens = re.findall(r"[A-Za-z0-9_]+", goal)
    found = []
    for t in tokens:
        m = fuzzy_match_token(t, name_choices, cutoff=0.6)
        if m and m not in found:
            found.append(m)
        if len(found) >= 2:
            break

    left, right = None, None
    m = re.search(r"([A-Za-z0-9_\-\s]+)\s+(vs|versus|v|vs\.|against)\s+([A-Za-z0-9_\-\s]+)", goal, flags=re.I)
    if m:
        a = m.group(1).strip()
        b = m.group(3).strip()
        ma = fuzzy_match_token(a, name_choices, cutoff=0.5)
        mb = fuzzy_match_token(b, name_choices, cutoff=0.5)
        if ma: left = ma
        if mb: right = mb

    if not left and found:
        left = found[0]
    if not right and len(found) >= 2:
        right = found[1]

    return left, right, subjects, intent

# -----------------------------
# Chart and pairwise helpers
# -----------------------------
def create_chart(df, chart_type, x, y, theme, color=None, size=None):
    title = f"{chart_type} of {y} vs {x}" + (f" (Colored by {color})" if color else "")
    try:
        if chart_type == "Bar":
            return px.bar(df, x=x, y=y, color=color if color else x, title=title, template=theme)
        if chart_type == "Line":
            return px.line(df, x=x, y=y, color=color, markers=True, title=title, template=theme)
        if chart_type == "Pie":
            return px.pie(df, names=x, values=y, hole=0.3, title=title, template=theme)
        if chart_type == "Area":
            return px.area(df, x=x, y=y, color=color, title=title, template=theme)
        if chart_type == "Scatter" or chart_type == "Bubble": 
            return px.scatter(df, x=x, y=y, color=color, size=size, hover_name=x, title=title, template=theme)
        if chart_type == "Histogram":
            return px.histogram(df, x=y, nbins=15, color=color, title=title, template=theme)
        if chart_type == "Treemap":
            return px.treemap(df, path=[x], values=y, title="Treemap: " + title, template=theme)
        if chart_type == "Sunburst":
            path = [p.strip() for p in x.split("+")] if "+" in x else [x]
            return px.sunburst(df, path=path, values=y, title="Sunburst: " + title, template=theme)
        if chart_type == "Stacked Bar":
            return px.bar(df, x=x, y=y, color=color, title=title, template=theme)
        # default
        return px.bar(df, x=x, y=y, title=title, template=theme)
    except Exception as e:
        # Fallback chart on error
        return px.bar(df, x=df.columns[0], y=df.columns[0] if len(df.columns) > 1 else None, title=f"Chart error: {e}", template=theme)

def create_pairwise_bar_and_radar(df, key_col, left_label, right_label, theme):
    left_rows = df[df[key_col].astype(str) == str(left_label)]
    right_rows = df[df[key_col].astype(str) == str(right_label)]
    if left_rows.empty or right_rows.empty:
        return None, None
    left_row = left_rows.iloc[0]
    right_row = right_rows.iloc[0]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return None, None

    comp_plot_df = pd.DataFrame({
        'metric': num_cols,
        str(left_label): [left_row[c] for c in num_cols],
        str(right_label): [right_row[c] for c in num_cols]
    })

    bar_fig = px.bar(comp_plot_df, x='metric', y=[str(left_label), str(right_label)], barmode='group',
                     title=f"{left_label} vs {right_label} — metrics comparison", template=theme)

    radar_df = pd.DataFrame({
        'metric': num_cols * 2,
        'value': [left_row[c] for c in num_cols] + [right_row[c] for c in num_cols],
        'entity': [str(left_label)] * len(num_cols) + [str(right_label)] * len(num_cols)
    })
    radar_fig = px.line_polar(radar_df, r='value', theta='metric', color='entity', line_close=True,
                              title=f"Radar: {left_label} vs {right_label}", template=theme)
    return bar_fig, radar_fig


def time_series_resample_and_forecast(df, date_col, value_col, freq='M', periods=3):
    if date_col not in df.columns or value_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return None, "Date column or value column invalid for time series analysis."
    
    ts_df = df.set_index(date_col)[[value_col]].resample(freq).sum().reset_index()
    ts_df = ts_df.dropna(subset=[value_col])

    if ts_df.shape[0] < 2:
        return None, "Not enough data points after resampling for trend analysis."

    X = np.arange(len(ts_df)).reshape(-1, 1)
    y_vals = ts_df[value_col].values
    
    slope, intercept = np.polyfit(X.flatten(), y_vals, 1)
    
    last_date = ts_df[date_col].iloc[-1]
    forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    forecast_indices = np.arange(len(ts_df), len(ts_df) + periods)
    
    forecast_values = slope * forecast_indices + intercept
    
    forecast_df = pd.DataFrame({
        date_col: forecast_dates,
        value_col: forecast_values
    })
    
    plot_df = pd.concat([ts_df, forecast_df], ignore_index=True)
    
    report = f"### 📈 Time Series Analysis ({freq} Resampling)\n"
    report += f"- Trend: **{slope:.2f}** per period (Intercept: {intercept:.2f})\n"
    report += f"- Last known {value_col}: **{ts_df[value_col].iloc[-1]:.2f}** on {last_date.strftime('%Y-%m-%d')}\n"
    report += f"- {periods} Period Forecast: **{forecast_df[value_col].iloc[-1]:.2f}** on {forecast_df[date_col].iloc[-1].strftime('%Y-%m-%d')}"
    
    return plot_df, report

# -----------------------------
# AI Summary & Suggestions (CONNECTED & POWERFUL)
# -----------------------------
def ai_insights_and_suggestions(df, x, y, chart_type, theme_choice, comparison_pair=None, selected_subjects=None, outlier_filter=None):
    
    # FIX: Initialize the local variable outlier_suggestion at the start
    outlier_suggestion = None 
    
    # --- Pairwise Analysis Logic (Retained) ---
    if comparison_pair:
        left_idx, right_idx = comparison_pair
        left_row = df_original.loc[df_original.index == left_idx]
        right_row = df_original.loc[df_original.index == right_idx]
        if left_row.empty: left_row = df_original.loc[df_original[df_original.columns[0]] == left_idx]
        if right_row.empty: right_row = df_original.loc[df_original[df_original.columns[0]] == right_idx]
        if left_row.empty or right_row.empty:
            return "Could not locate both entities for comparison.", ["Check pairwise selectors or uploaded file key column."]

        left_row = left_row.iloc[0]
        right_row = right_row.iloc[0]
        num_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in selected_subjects if c in num_cols] if selected_subjects else num_cols
        
        lines = [f"### ⚔️ Pairwise Analysis — {left_row[df_original.columns[0]]} vs {right_row[df_original.columns[0]]}"]
        for c in cols:
            l, r = left_row[c], right_row[c]
            diff = l - r
            pct = (diff / r * 100) if r != 0 else None
            pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
            trend = "higher" if diff > 0 else ("lower" if diff < 0 else "equal")
            lines.append(f"- **{c}**: {left_row[df_original.columns[0]]} = **{l}**, {right_row[df_original.columns[0]]} = **{r}** → {left_row[df_original.columns[0]]} is {trend} ({pct_str}).")
        
        left_sum = sum([left_row[c] for c in cols])
        right_sum = sum([right_row[c] for c in cols])
        leader = left_row[df_original.columns[0]] if left_sum > right_sum else right_row[df_original.columns[0]] if right_sum > left_sum else "Tie"
        lines.append(f"\n**Summary:** Total across compared metrics — {left_row[df_original.columns[0]]}: **{left_sum}**, {right_row[df_original.columns[0]]}: **{right_sum}**. Leader: **{leader}**.")
        suggestions = ["Highlight specific subjects by typing e.g. 'Compare A vs B in Math and Science'.", "Use radar chart for visual multi-dimension comparison (available below the pairwise table).", "Export the pairwise table as CSV for reporting."]
        return "\n".join(lines), suggestions

    # --- Dataset-level insights (ENHANCED) ---
    lines = []
    lines.append("### 🧩 Dataset Summary")
    if outlier_filter:
        filtered_count = df_original.shape[0] - df.shape[0] 
        lines.append(f"**Filter Active:** Removing **{filtered_count}** row(s) based on {outlier_filter[0]['column']} outliers.")
    lines.append(f"- Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    if num_cols: lines.append(f"- Numeric fields: {', '.join(num_cols)}")
    if cat_cols: lines.append(f"- Categorical fields: {', '.join(cat_cols)}")
    if date_cols: lines.append(f"- Time Series fields: {', '.join(date_cols)}")

    # 1. Trend Interpretation & Outlier Check (for y-axis)
    if y in df.columns and y in num_cols:
        lines.append(f"\n### 📊 `{y}` Stats & Trend Analysis")
        desc = df[y].describe()
        lines.append(f"- Key Metrics: Mean **{desc['mean']:.2f}**, Std Dev **{desc['std']:.2f}**, Skewness **{df[y].skew():.2f}**.")
        
        # Trend over X if X is numeric or date
        if x in num_cols or x in date_cols:
            df_temp = df[[x, y]].dropna()
            if not df_temp.empty and df_temp.shape[0] > 1:
                if x in date_cols:
                    X_trend = np.arange(len(df_temp)).reshape(-1, 1)
                else:
                    X_trend = df_temp[x].values.reshape(-1, 1)

                y_trend = df_temp[y].values
                try:
                    slope, _ = np.polyfit(X_trend.flatten(), y_trend, 1)
                    trend_str = "increasing significantly (strong positive correlation)" if slope > (df[y].std() / 5) else \
                                "decreasing significantly (strong negative correlation)" if slope < -(df[y].std() / 5) else \
                                "mostly stable"
                    lines.append(f"- **Overall Trend:** `{y}` is **{trend_str}** with respect to `{x}` (Slope: {slope:.2f}).")
                except Exception:
                    pass
        
        # Outlier Detection
        Z = np.abs((df[y] - df[y].mean()) / df[y].std())
        outliers = df[Z > 3]
        if not outliers.empty:
            lines.append(f"**⚠️ Outlier Alert:** {len(outliers)} rows in `{y}` have Z-score > 3. (Min: {outliers[y].min():.2f}, Max: {outliers[y].max():.2f}).")
            outlier_suggestion = f"Apply filter to exclude {y} outliers (Z > 3)."

    # 2. Categorical Dominance Analysis
    if x in cat_cols and y in num_cols:
        agg = df.groupby(x)[y].sum().sort_values(ascending=False)
        if not agg.empty:
            total = agg.sum()
            top_category = agg.index[0]
            top_share = agg.iloc[0] / total * 100
            
            lines.append(f"\n### 🗺️ Categorical Dominance: `{x}` vs `{y}`")
            lines.append(f"- **Dominant Factor:** **{top_category}** contributes **{top_share:.1f}%** of the total `{y}`.")
            if top_share > 50:
                 lines.append(f"- **Concentration:** High concentration suggests **{top_category}** is the primary driver, masking performance in other categories. Drill down may be required.")
            else:
                 lines.append(f"- **Diversity:** Contribution is spread across categories. No single factor dominates the total `{y}`.")
                 

    # 3. Cross-Correlation Insights (ENHANCED)
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        us = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
        cnt = 0
        lines.append("\n### 🔗 Correlation Highlights")
        
        correlation_insights = []
        for (a,b), val in us.items():
            if a == b or val == 1 or cnt >= 3: continue
            
            r = corr.loc[a,b]
            strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
            direction = "positive" if r > 0 else "negative"
            
            if 'Math' in a and 'Science' in b or 'Science' in a and 'Math' in b:
                context = "This suggests a student's aptitude often translates across technical subjects."
            elif 'Sales' in a and 'Profit' in b or 'Profit' in a and 'Sales' in b:
                context = "This expected relationship validates the overall financial health of the data."
            elif 'Hours Studied' in a and 'Score' in b:
                context = "This reinforces the intuitive idea that study effort is a direct performance factor."
            else:
                context = "Investigate potential causality or lurking variables between these two metrics."

            correlation_insights.append(f"- **{a}** vs **{b}** : **{r:.2f}** ({strength} {direction}). *Insight:* {context}")
            cnt += 1
            
        lines.extend(correlation_insights)

    lines.append("\n---\n**🔍 Conclusion:** Data insights generated. Use visual filters or drill-downs to investigate drivers and outliers.")

    suggestions = [
        f"Try '{'Line' if chart_type=='Line' else 'Bar'}' charts for {chart_type.lower()} visualizations.",
        "Use 'Distribution' intent or histogram to inspect spread/outliers.",
        "Select specific subjects in the sidebar or use natural language to focus analysis."
    ]
    
    # This check is now safe due to the initialization at the start of the function.
    if outlier_suggestion and outlier_suggestion not in suggestions:
        suggestions.insert(0, outlier_suggestion)

    return "\n".join(lines), suggestions

# -----------------------------
# UI: Sidebar (controls)
# -----------------------------
st.sidebar.title("⚙️ Controls")

examples = [
    "Compare A vs B in Math and Hours Studied",
    "Show monthly Sales trend and 3-month forecast",
    "Analyze relationship between Sales and Profit",
    "Show distribution of Math scores",
    "Compare North vs South"
]
goal = st.sidebar.text_area("🧠 Your Data Goal", placeholder=np.random.choice(examples), height=80)

dataset_name = st.sidebar.selectbox("📂 Choose Dataset", list(DATASETS.keys()))
theme_choice = st.sidebar.radio("🎨 Theme", list(THEMES.keys()), horizontal=True)
chart_choice = st.sidebar.selectbox("📈 Chart Type", ["Auto", "Bar", "Line", "Pie", "Area", "Scatter", "Bubble", "Histogram", "Treemap", "Sunburst", "Stacked Bar"])
palette_choice = st.sidebar.selectbox("🎨 Palette", PALETTES)
uploaded_file = st.sidebar.file_uploader("⬆️ Upload CSV/Excel", type=["csv", "xlsx"])

# --- Separator ---
st.sidebar.markdown("---")
st.sidebar.caption("💡 Made with ❤️ by Team Innovex Coders")

# -----------------------------
# LOAD DATA & PRE-PROCESSING
# -----------------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")
        df = DATASETS[dataset_name]
else:
    df = DATASETS[dataset_name]

df = normalize_cols(df)
df_original = df.copy() # Keep a copy of original data

possible_keys = [c for c in df.columns if any(k in c.lower() for k in ['student', 'name', 'id', 'label', 'entity', 'region'])]
key_col = possible_keys[0] if possible_keys else df.columns[0]

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()
date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

# -----------------------------
# INTERPRET GOAL AND DEFINE FINAL CONTROLS
# -----------------------------
name_candidates = df[key_col].astype(str).unique().tolist() if key_col in df.columns else []
subject_candidates = num_cols

# parse goal
parsed_left, parsed_right, parsed_subjects, parsed_intent = parse_goal(goal or "", name_candidates, subject_candidates)

# --- AI-DRIVEN DEFAULT AXIS SELECTION ---
default_x, default_y, default_color, default_size = None, None, None, None
x_choices = (cat_cols + num_cols + date_cols) if (cat_cols + num_cols + date_cols) else df.columns.tolist()

if not x_choices:
    x_choices = df.columns.tolist()
    
if num_cols:
    default_y = num_cols[0]
    
    if parsed_intent in ["trend", "forecast"] and date_cols:
        default_x = date_cols[0]
        default_y = parsed_subjects[0] if parsed_subjects and parsed_subjects[0] in num_cols else default_y
    elif parsed_intent == "comparison" or parsed_intent == "composition":
        default_x = key_col
        default_y = parsed_subjects[0] if parsed_subjects and parsed_subjects[0] in num_cols else default_y
    elif parsed_intent == "distribution":
        default_x = key_col if key_col in x_choices else x_choices[0]
        default_y = parsed_subjects[0] if parsed_subjects and parsed_subjects[0] in num_cols else default_y
    elif parsed_intent == "relationship":
        default_x = num_cols[0] if len(num_cols) >= 1 else x_choices[0]
        default_y = num_cols[1] if len(num_cols) >= 2 else num_cols[0]
        default_color = cat_cols[0] if cat_cols else None
        default_size = num_cols[2] if len(num_cols) >= 3 else None
    else: # Default (Insight)
        default_x = key_col if key_col in x_choices else x_choices[0]
        default_y = parsed_subjects[0] if parsed_subjects and parsed_subjects[0] in num_cols else default_y
else:
    default_x = x_choices[0]
    default_y = x_choices[0]

# --- Determine Index for Selectboxes ---
def get_index(choice, options):
    try:
        if choice and choice in options:
            return options.index(choice)
        return 0
    except ValueError:
        return 0
        
x_index = get_index(default_x, x_choices)
y_index = get_index(default_y, num_cols) if num_cols else 0


# --- Manual Pairwise Selectors (Retaining original layout) ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔎 Pairwise Comparison")
select_left = st.sidebar.selectbox("Left (A)", options=[None] + name_candidates, index=0)
select_right = st.sidebar.selectbox("Right (B)", options=[None] + name_candidates, index=0)
st.sidebar.markdown("---")

left_name = select_left if select_left else parsed_left
right_name = select_right if select_right else parsed_right

# --- Main Axis Selectors (AI-Driven defaults) ---
st.sidebar.markdown("## Axis & Chart")
x = st.sidebar.selectbox("X-Axis / Category / Date", options=x_choices, index=x_index)
y = st.sidebar.selectbox("Y-Axis / Numeric", options=num_cols, index=y_index) if num_cols else x

# --- Multivariate Controls (AI-Driven defaults) ---
st.sidebar.markdown("### 🌈 Multivariate Attributes")
color_options = [None] + cat_cols + num_cols
size_options = [None] + num_cols

color_index = get_index(default_color, color_options)
size_index = get_index(default_size, size_options)

color_by = st.sidebar.selectbox("Color By", options=color_options, index=color_index)
size_by = st.sidebar.selectbox("Size By (Bubble/Scatter)", options=size_options, index=size_index)

# Determine chart suggestion by parsed_intent (unchanged logic)
chart_suggestion = "Bar"
if parsed_intent == "trend" or parsed_intent == "forecast": chart_suggestion = "Line"
elif parsed_intent == "distribution": chart_suggestion = "Histogram"
elif parsed_intent == "composition": chart_suggestion = "Treemap"
elif parsed_intent == "relationship": 
    chart_suggestion = "Scatter"
    if size_by: chart_suggestion = "Bubble"

final_chart = chart_choice if chart_choice != "Auto" else chart_suggestion
if final_chart == "Bubble": final_chart = "Scatter"
theme = THEMES[theme_choice]


# -----------------------------
# AI-DRIVEN FILTERING
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## 🧹 Data Filtering [AI]")
filter_outliers = st.sidebar.checkbox(f"Exclude Outliers in `{y}` (Z > 3)", value=False)
outlier_rows = []

if filter_outliers and y in num_cols:
    Z = np.abs((df[y] - df[y].mean()) / df[y].std())
    outlier_rows = df[Z > 3].index.tolist()
    if outlier_rows:
        df = df.drop(outlier_rows)
        st.sidebar.info(f"Filtered out {len(outlier_rows)} outlier row(s) in `{y}`.")

active_filter_info = [{"column": y, "count": len(outlier_rows)}] if outlier_rows else None

# -----------------------------
# MAIN layout
# -----------------------------
st.markdown("# 💡 VIBE-CODE AI STUDIO ULTRA — Powerful AI Summary")
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.markdown("### Quick Overview")
    st.write(f"**Dataset:** {uploaded_file.name if uploaded_file else dataset_name}")
    st.write(f"**Rows (Filtered):** {df.shape[0]} | **Columns:** {df.shape[1]}")
with col2:
    st.metric("Numeric fields", len(num_cols))
with col3:
    st.metric("Date/Categorical fields", len(cat_cols) + len(date_cols))

st.markdown("---")

# Data overview
st.markdown("## 📊 Data Overview")
with st.expander("Preview data (first 10 rows)"):
    st.dataframe(df.head(10))
with st.expander("Summary statistics"):
    st.dataframe(df.describe(include='all'))

# -----------------------------
# TIME SERIES / FORECASTING SECTION
# -----------------------------
if (parsed_intent == "trend" or parsed_intent == "forecast") and x in date_cols and y in num_cols:
    st.markdown("## ⏳ Time Series & Trend Analysis")
    date_col = x
    col_resample, col_forecast = st.columns(2)
    
    with col_resample:
        resample_freq = st.selectbox("Resampling Frequency", options=['D', 'W', 'M', 'Q', 'Y'], index=2, key='freq')
    with col_forecast:
        forecast_periods = st.slider("Forecast Periods", min_value=0, max_value=12, value=3, key='periods')

    ts_plot_df, ts_report = time_series_resample_and_forecast(df, date_col, y, freq=resample_freq, periods=forecast_periods)
    
    if ts_plot_df is not None:
        st.markdown(ts_report)
        
        fig_ts = px.line(ts_plot_df, x=date_col, y=y, title=f"Trend of {y} ({resample_freq} Resample)", template=theme)
        
        if forecast_periods > 0:
            forecast_start_index = ts_plot_df.shape[0] - forecast_periods
            fig_ts.add_scatter(x=ts_plot_df[date_col].iloc[forecast_start_index:], y=ts_plot_df[y].iloc[forecast_start_index:],
                               mode='lines', name='Forecast', line=dict(dash='dash', color='red'))
        
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
         st.warning(ts_report)
    st.markdown("---")


# -----------------------------
# Render dataset-level chart (Main Chart)
# -----------------------------
st.markdown(f"## 🎯 Visualization — {final_chart} Chart ({parsed_intent.capitalize()} Intent)")
try:
    fig = create_chart(df, final_chart, x, y, theme, color=color_by, size=size_by)
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Could not render chart: {e}")

# -----------------------------
# Pairwise comparison logic
# -----------------------------
compare_pair = None
if left_name and right_name and left_name != right_name:
    left_idx_list = df_original.index[df_original[key_col].astype(str) == str(left_name)].tolist()
    right_idx_list = df_original.index[df_original[key_col].astype(str) == str(right_name)].tolist()
    if left_idx_list and right_idx_list:
        compare_pair = (left_idx_list[0], right_idx_list[0])

if compare_pair:
    st.markdown("## ⚔️ Pairwise Comparison")
    lidx, ridx = compare_pair
    left_row = df_original.loc[lidx]
    right_row = df_original.loc[ridx]
    comp_df = pd.DataFrame({str(left_name): left_row, str(right_name): right_row})
    st.dataframe(comp_df)

    try:
        bar_fig, radar_fig = create_pairwise_bar_and_radar(df_original, key_col, left_name, right_name, theme)
        col_bar, col_radar = st.columns(2)
        with col_bar:
            if bar_fig: st.plotly_chart(bar_fig, use_container_width=True)
        with col_radar:
            if radar_fig: st.plotly_chart(radar_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render pairwise charts: {e}")

# -----------------------------
# AI Analytical Report & Suggestions
# -----------------------------
st.markdown("## 🧠 AI Analytical Report")
report_md, suggestions = ai_insights_and_suggestions(df, x, y, final_chart, theme_choice, comparison_pair=compare_pair if compare_pair else None, selected_subjects=parsed_subjects, outlier_filter=active_filter_info)
st.markdown(report_md)

# Show suggestions under AI report in the same section
if suggestions:
    with st.expander("💡 Suggestions & Next Steps", expanded=True):
        for s in suggestions:
            st.write("- " + s)

# -----------------------------
# Extra analytics: Heatmap (same area)
# -----------------------------
if len(num_cols) > 1:
    st.markdown("## 🔥 Correlation Heatmap")
    try:
        corr_fig = px.imshow(df[num_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r",
                             title="Numeric Correlations", template=theme)
        st.plotly_chart(corr_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render correlation heatmap: {e}")

# -----------------------------
# Export / Download (fixed)
# -----------------------------
export_col1, export_col2, export_col3 = st.columns(3)
with export_col1:
    st.caption("Download Original Data")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_original.to_excel(writer, index=False, sheet_name='original_data')
    buffer.seek(0)
    st.download_button(label="📥 Download Excel (Original)", data=buffer.getvalue(), file_name="vibe_code_original.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with export_col2:
    st.caption("Download Filtered Data")
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download CSV (Filtered)", data=csv_bytes, file_name="vibe_code_filtered.csv", mime='text/csv')

# -----------------------------
# Footer (unchanged)
# -----------------------------
st.markdown("---")
st.markdown("<center>✨ Made with ❤️ by <b>Team Innovex Coders</b> — Powerful AI Summary Edition | Powered by Streamlit + Plotly</center>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.info("Tip: The AI now automatically sets the X/Y axes and chart type based on your goal!")
