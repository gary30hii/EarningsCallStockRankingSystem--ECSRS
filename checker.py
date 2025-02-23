import json


def check_speaker1(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)  # Load JSON data from file

        for i, entry in enumerate(data):
            if "speaker1" not in entry:
                print(f"Entry {i + 1}: 'speaker1' key is missing.")
            elif entry["speaker1"] == "":
                print(f"Entry {i + 1}: 'speaker1' key exists but is empty.")
        print("Check completed.")

    except Exception as e:
        print(f"Error reading or processing file: {e}")


# Input your file path here
file_path = "CRWD/earnings_transcripts_2020_2024.json"
check_speaker1(file_path)
