import streamlit as st
import pandas as pd
import torch
import json
import joblib
import sqlite3
from datetime import datetime
from transformers import pipeline

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipeline(
    "text-classification", model="ProsusAI/finbert", device=0 if device == "mps" else -1
)


def init_db():
    conn = sqlite3.connect("ecsrs_results.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            actual_eps REAL,
            estimated_eps REAL,
            firm_size REAL,
            tone_score REAL,
            predicted_category TEXT,
            confidence_json TEXT,
            transcript_json TEXT
        )
    """
    )
    conn.commit()
    return conn


def split_text(text, chunk_size=250, overlap=50):
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]


def get_sentiment_score(text):
    chunks = split_text(text)
    total_words = len(text.split())
    sentiment_scores = []
    for chunk in chunks:
        try:
            if len(chunk.split()) > 512:
                chunk = " ".join(chunk.split()[:512])
            result = pipe(chunk)[0]
            label, score = result["label"], result["score"]
            word_count = len(chunk.split())
            sentiment_scores.append(
                score * word_count
                if label == "positive"
                else -score * word_count if label == "negative" else 0
            )
        except Exception as e:
            st.warning(f"Error processing chunk: {e}")
    if total_words == 0:
        return 0, 0
    return sum(sentiment_scores) / total_words, total_words


def get_weighted_sentiment(speakers):
    total_words = 0
    weighted_score_sum = 0
    for speaker_key, speaker_data in speakers.items():
        text = speaker_data["content"]
        score, word_count = get_sentiment_score(text)
        total_words += word_count
        weighted_score_sum += score * word_count
    return weighted_score_sum / total_words if total_words > 0 else None


def calculate_earnings_surprise(actual, estimated):
    return (actual - estimated) / estimated if estimated != 0 else None


def run_prediction(actual, estimated, size, transcript):
    earnings_surprise = calculate_earnings_surprise(actual, estimated)
    tone = get_weighted_sentiment(transcript)
    model = joblib.load("random_forest_best_model.pkl")
    new_data = pd.DataFrame(
        [[earnings_surprise, size, tone]],
        columns=["Earnings_Surprise", "Firm_Size", "method_2"],
    )
    predicted_category = model.predict(new_data)[0]
    confidence_scores = model.predict_proba(new_data)[0]
    category_labels = model.classes_
    confidence_dict = {
        category_labels[i]: confidence_scores[i] for i in range(len(category_labels))
    }
    return predicted_category, confidence_dict, tone


def save_prediction_to_db(
    conn, actual, estimated, size, tone, category, confidence, speakers
):
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO predictions (
            timestamp, actual_eps, estimated_eps, firm_size,
            tone_score, predicted_category, confidence_json, transcript_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            actual,
            estimated,
            size,
            tone,
            category,
            json.dumps(confidence),
            json.dumps(speakers),
        ),
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────
conn = init_db()
st.set_page_config(page_title="ECSRS: Earnings Call Rank Predictor", layout="wide")
st.title("📈 Earnings Call Stock Ranking System (ECSRS)")

st.sidebar.header("Model Inputs")
actual_earnings = st.sidebar.number_input("Actual EPS", value=1.25)
estimated_earnings = st.sidebar.number_input("Estimated EPS", value=0.94)
firm_size = st.sidebar.number_input("Firm Size (Market Cap)", value=6_089_250_000)

st.markdown("### 👥 Enter Speaker Information")
num_speakers = st.number_input(
    "Number of Speakers", min_value=1, max_value=10, value=2, step=1
)
speaker_data = {}
for i in range(1, num_speakers + 1):
    with st.expander(f"Speaker {i}", expanded=True):
        name = st.text_input(f"Speaker {i} Name", key=f"name_{i}")
        content = st.text_area(f"Speaker {i} Content", height=300, key=f"content_{i}")
        speaker_data[f"speaker{i}"] = {"name": name, "content": content}

if st.button("🚀 Run Prediction"):
    category, confidence, tone = run_prediction(
        actual_earnings, estimated_earnings, firm_size, speaker_data
    )
    save_prediction_to_db(
        conn,
        actual_earnings,
        estimated_earnings,
        firm_size,
        tone,
        category,
        confidence,
        speaker_data,
    )
    st.success(f"🎯 Predicted Rank Category: `{category}`")
    st.markdown(f"**🧠 Tone Sentiment Score:** `{tone:.4f}`")
    st.markdown("#### 🔍 Confidence Scores")
    st.json(confidence)
    st.markdown("#### 📄 Full Structured JSON")
    st.json(speaker_data)

# ─────────────────────────────────────────────────────────────
# HISTORY VIEW & TOOLS
# ─────────────────────────────────────────────────────────────
st.markdown("## 📊 Past Prediction History")

st.markdown("### 📅 Filter by Date Range")
start_date = st.date_input("Start Date", datetime.today().replace(day=1))
end_date = st.date_input("End Date", datetime.today())
start_dt = start_date.strftime("%Y-%m-%d 00:00:00")
end_dt = end_date.strftime("%Y-%m-%d 23:59:59")

c = conn.cursor()
c.execute(
    """
    SELECT id, timestamp, actual_eps, estimated_eps, firm_size, tone_score, predicted_category 
    FROM predictions WHERE timestamp BETWEEN ? AND ? ORDER BY id DESC
""",
    (start_dt, end_dt),
)
rows = c.fetchall()
cols = [
    "ID",
    "Timestamp",
    "Actual EPS",
    "Estimated EPS",
    "Firm Size",
    "Tone",
    "Category",
]
df = pd.DataFrame(rows, columns=cols)

if not df.empty:
    st.dataframe(df)
    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Full Prediction History as CSV",
        csv_export,
        file_name="ecsrs_predictions.csv",
        mime="text/csv",
    )

    with st.expander("⚠️ Admin Tools"):
        if st.button("🗑️ Delete ALL Records"):
            c.execute("DELETE FROM predictions")
            conn.commit()
            st.warning("All prediction records deleted. Refresh to update view.")

    selected_id = st.selectbox("Select a prediction ID to view details", df["ID"])
    if st.button("🔍 View Selected Prediction"):
        c.execute(
            "SELECT confidence_json, transcript_json FROM predictions WHERE id = ?",
            (selected_id,),
        )
        confidence_json, transcript_json = c.fetchone()
        st.markdown("### 🎯 Confidence Scores")
        st.json(json.loads(confidence_json))
        st.markdown("### 📄 Transcript")
        st.json(json.loads(transcript_json))
else:
    st.info("No predictions stored yet.")
