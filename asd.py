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


def get_weighted_sentiment(speakers):
    """Computes the overall weighted sentiment score from multiple speakers."""
    total_words = 0
    weighted_score_sum = 0

    for speaker_key, speaker_data in speakers.items():
        speaker_name = speaker_data.get("name", speaker_key)
        text = speaker_data["content"]

        print(f"\nProcessing {speaker_name}...")

        score, word_count = get_sentiment_score(text)

        # Accumulate total words and weighted score sum
        total_words += word_count
        weighted_score_sum += score * word_count

    # Compute final weighted sentiment score for the entire dataset
    if total_words > 0:
        return weighted_score_sum / total_words

    return None  # If no valid data


if __name__ == "__main__":
    # Sample input formatted with speaker keys
    data = {
        "speaker1": {
            "name": "John Doe",
            "content": "We had a weak quarter.",
        },
        "speaker2": {
            "name": "Jane Smith",
            "content": "Our financials remain stable despite challenges.",
        },
        "speaker3": {
            "name": "Alice Johnson",
            "content": "Market conditions were tough, but we adapted.",
        },
    }

    overall_score = get_weighted_sentiment(data)
    print(
        f"Overall Weighted Sentiment Score: {overall_score:.4f}"
        if overall_score is not None
        else "No valid sentiment data."
    )


def calculate_earnings_surprise(actual, estimated):
    """Calculate earnings surprise percentage."""
    if estimated == 0:
        return None  # Avoid division by zero
    return (actual - estimated) / estimated
