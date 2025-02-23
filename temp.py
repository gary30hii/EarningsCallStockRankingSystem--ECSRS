import os
import json
import pandas as pd
import multiprocessing
import ollama
import re
import textwrap

# Disable Hugging Face tokenizers parallelism to avoid multiprocessing conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set multiprocessing start method for macOS
multiprocessing.set_start_method("spawn", force=True)

# Model name (Ensure it’s correct)
MODEL_NAME = "llama3.2:latest"  # Update this if necessary


def analyze_sentiment(text, max_words=500):
    """
    Analyzes the sentiment of a given text using Ollama's LLaMA model.

    Args:
        text (str): The input text to analyze.
        max_words (int): Maximum words per chunk before splitting (default: 50).

    Returns:
        tuple: (overall_sentiment, total_words)
               - overall_sentiment (float): Average sentiment score (-1 to 1).
               - total_words (int): Total word count of the input text.
    """

    if not text.strip():
        return 0.0, 0

    def get_sentiment_score(chunk):
        """Gets the sentiment score for a text chunk."""
        prompt = f"""
        Analyze the tone of management in the following earnings call statement and return a score between -1.0 (very negative) and 1.0 (very positive).

        Scoring Guide:
        - Positive (0.5 to 1.0): Confident, optimistic, strong leadership, clear vision.
        - Neutral (-0.4 to 0.4): Balanced, factual, cautious, diplomatic.
        - Negative (-1.0 to -0.5): Uncertain, hesitant, defensive, concerned.

        Focus on language and tone, not financial numbers.

        Text:
        {chunk}

        Tone Score (only a number):
        """

        try:
            response = ollama.chat(
                model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
            )
            score_text = response["message"]["content"].strip()
            match = re.search(r"-?\d+(?:\.\d+)?", score_text)
            return float(match.group()) if match else 0.0
        except Exception as e:
            print(f"Error: {e}")
            return 0.0  # Default to neutral if API fails

    def split_text(text, max_words=500):
        """Splits text into smaller chunks while retaining sentence structure."""
        words = text.split()
        return textwrap.wrap(" ".join(words), width=max_words)

    # Step 1: Split text into smaller chunks
    text_chunks = split_text(text, max_words=max_words)

    # Step 2: Analyze sentiment for each chunk
    sentiment_scores = [get_sentiment_score(chunk) for chunk in text_chunks]

    # Step 3: Compute the overall sentiment score
    overall_sentiment = (
        sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    )
    total_words = len(text.split())

    return overall_sentiment, total_words  # Return as a tuple


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
                score, word_count = analyze_sentiment(text)

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


def process_car_after_earnings(folder_path, sentiment_df, processed_files_count):
    """Merge sentiment scores into CAR_after_earnings.csv and count processed files."""
    car_file_path = os.path.join(folder_path, "CAR_after_earnings.csv")

    if os.path.exists(car_file_path):
        car_df = pd.read_csv(car_file_path)

        # Check if 'method_3' column already exists
        if "method_3" in car_df.columns:
            print(
                f"Skipping sentiment analysis for {car_file_path} as 'method_3' already exists."
            )
            return processed_files_count + 1

        if "Earnings_Call_Date" not in car_df.columns:
            print(f"Error: 'Earnings_Call_Date' column not found in {car_file_path}")
            return processed_files_count

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
        merged_df.rename(columns={"overall_sentiment_score": "method_3"}, inplace=True)

        merged_df.to_csv(car_file_path, index=False)
        print(f"Updated {car_file_path} with sentiment scores.")

        # Increment the count of processed files
        return processed_files_count + 1
    else:
        print(f"No CAR_after_earnings.csv found in {folder_path}")
        return processed_files_count


def load_json_files_from_subdirectories():
    """Find and process earnings_transcripts_2020_2024.json in each subdirectory."""
    current_directory = os.getcwd()
    processed_files_count = 0  # Tracks total processed files
    already_has_method_3_count = 0  # Tracks files that already had 'method_3'
    newly_added_method_3_count = 0  # Tracks files where 'method_3' was added

    for subdir in os.listdir(current_directory):
        subdir_path = os.path.join(current_directory, subdir)
        json_file_path = os.path.join(
            subdir_path, "earnings_transcripts_2020_2024.json"
        )
        car_file_path = os.path.join(subdir_path, "CAR_after_earnings.csv")

        if os.path.isdir(subdir_path) and os.path.exists(json_file_path):
            # First check if CAR_after_earnings.csv exists
            if not os.path.exists(car_file_path):
                print(f"Skipping {subdir}: No CAR_after_earnings.csv found.")
                continue  # Skip to the next directory

            # Read CAR_after_earnings.csv to check if 'method_3' already exists
            car_df = pd.read_csv(car_file_path)
            if "method_3" in car_df.columns:
                print(f"Skipping {car_file_path}: 'method_3' already exists.")
                already_has_method_3_count += 1  # Count it as processed
                processed_files_count += 1
                continue  # Skip further processing

            # If 'method_3' is not present, analyze and add sentiment scores
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                sentiment_results = analyze_weighted_sentiment(data)
                process_car_after_earnings(subdir_path, sentiment_results, 0)

                newly_added_method_3_count += 1
                processed_files_count += 1

            # Print summary of processed files
            print("\nSummary of processed files:")
            print(f" - Total files processed: {processed_files_count}")
            print(
                f" - Files where 'method_3' was already present: {already_has_method_3_count}"
            )
            print(
                f" - Files where 'method_3' was newly added: {newly_added_method_3_count}"
            )


if __name__ == "__main__":
    load_json_files_from_subdirectories()
