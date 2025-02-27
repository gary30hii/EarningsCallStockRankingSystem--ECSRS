import os
import json
import pandas as pd
import multiprocessing
import torch
from transformers import pipeline

# Disable Hugging Face tokenizers parallelism to avoid multiprocessing conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set multiprocessing start method for macOS
multiprocessing.set_start_method("spawn", force=True)

# Check if MPS (Metal Performance Shaders) is available for acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load FinBERT sentiment analysis pipeline with MPS acceleration
pipe = pipeline(
    "text-classification", model="ProsusAI/finbert", device=0 if device == "mps" else -1
)


def split_text(text, chunk_size=250, overlap=50):
    """Splits long text into overlapping chunks to fit FinBERT's token limit."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def get_sentiment_score(text):
    """Return the sentiment score (-1 to 1) using ProsusAI/FinBERT, handling long text."""
    chunks = split_text(text)  # Get chunks
    total_words = len(text.split())  # Original total word count
    sentiment_scores = []

    for i, chunk in enumerate(chunks):
        try:
            if len(chunk.split()) > 512:
                chunk = " ".join(
                    chunk.split()[:512]
                )  # Ensure chunk does not exceed model limit
            result = pipe(chunk)[0]  # Run chunk through FinBERT model
            label = result["label"]
            score = result["score"]
            word_count = len(chunk.split())

            if label == "positive":
                sentiment_scores.append(score * word_count)
            elif label == "negative":
                sentiment_scores.append(-score * word_count)
            else:
                sentiment_scores.append(0)

            print(
                f"Chunk {i+1}: Score: {score:.4f}, Words: {word_count}, Label: {label}"
            )
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")

    if total_words == 0:
        return 0, total_words  # Avoid division by zero

    overall_sentiment = (
        sum(sentiment_scores) / total_words
    )  # Weighted average sentiment
    return overall_sentiment, total_words


def analyze_weighted_sentiment(json_data):
    """Analyze and compute the weighted overall sentiment score per quarter."""
    quarterly_scores = {}
    detailed_results = []

    for entry in json_data:
        raw_date = entry["date"]
        date = pd.to_datetime(raw_date).strftime("%Y-%m-%d")

        total_words_in_quarter = 0
        weighted_score_sum = 0

        for key, speaker_data in entry.items():
            if isinstance(speaker_data, dict) and "content" in speaker_data:
                speaker_name = speaker_data.get("name", key)
                text = speaker_data["content"]
                print(f"\nProcessing {speaker_name} on {date}...")
                score, word_count = get_sentiment_score(text)

                # Store detailed results for each speaker
                detailed_results.append(
                    {
                        "date": date,
                        "key": key,
                        "name": speaker_name,
                        "total_words": word_count,
                        "score": score,
                    }
                )

                # Update quarter-level sentiment score
                total_words_in_quarter += word_count
                weighted_score_sum += score * word_count

        if total_words_in_quarter > 0:
            quarterly_scores[date] = weighted_score_sum / total_words_in_quarter

    # Convert to DataFrame and sort by date in ascending order
    sentiment_df = pd.DataFrame(
        quarterly_scores.items(), columns=["date", "overall_sentiment_score"]
    ).sort_values(by="date", ascending=True)

    return sentiment_df


def process_car_after_earnings(folder_path, sentiment_df):
    """Merge sentiment scores into CAR_after_earnings.csv and save the updated file."""
    car_file_path = os.path.join(folder_path, "CAR_after_earnings.csv")

    if os.path.exists(car_file_path):
        car_df = pd.read_csv(car_file_path)

        if "Earnings_Call_Date" not in car_df.columns:
            print(f"Error: 'Earnings_Call_Date' column not found in {car_file_path}")
            return

        car_df["Earnings_Call_Date"] = pd.to_datetime(
            car_df["Earnings_Call_Date"]
        ).dt.date
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date

        sentiment_df = sentiment_df.groupby("date", as_index=False)[
            "overall_sentiment_score"
        ].mean()

        if car_df["Earnings_Call_Date"].duplicated().any():
            print("\nWarning: Duplicates found in CAR data. Aggregating...")
            car_df = car_df.groupby("Earnings_Call_Date", as_index=False).mean()

        merged_df = car_df.merge(
            sentiment_df, left_on="Earnings_Call_Date", right_on="date", how="left"
        )

        merged_df.drop(columns=["date"], inplace=True)
        merged_df.rename(columns={"overall_sentiment_score": "method_2"}, inplace=True)

        merged_df.to_csv(car_file_path, index=False)
        print(f"Updated {car_file_path} with sentiment scores.")
    else:
        print(f"No CAR_after_earnings.csv found in {folder_path}")


def load_json_files_from_subdirectories():
    """Find and process earnings_transcripts_2020_2024.json in each subdirectory."""
    current_directory = os.getcwd()
    for subdir in os.listdir(current_directory):
        subdir_path = os.path.join(current_directory, subdir)
        json_file_path = os.path.join(
            subdir_path, "earnings_transcripts_2020_2024.json"
        )

        if os.path.isdir(subdir_path) and os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                sentiment_results = analyze_weighted_sentiment(data)
                process_car_after_earnings(subdir_path, sentiment_results)


if __name__ == "__main__":
    load_json_files_from_subdirectories()
