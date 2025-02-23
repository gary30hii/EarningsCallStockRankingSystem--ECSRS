import os
import json
import re
import unicodedata


# Function to normalize speaker names
def normalize_speaker_name(name):
    """Normalize speaker names: lowercase, remove extra spaces, allow underscores."""
    name = name.lower().strip()
    name = re.sub(r"\s+", "_", name)  # Replace spaces with underscores
    return re.sub(
        r"[^a-z0-9_]", "", name
    )  # Remove non-alphanumeric characters except underscores


# Preprocess transcript to normalize speaker formatting
def preprocess_transcript(transcript):
    # Convert Unicode escape sequences to proper characters
    transcript = unicodedata.normalize("NFKC", transcript)

    # Replace curly apostrophes with standard ones
    transcript = transcript.replace("\u2019", "'")

    # Normalize cases where there is a newline and a space before ":"
    transcript = re.sub(r"([\w\s.,;&'\\-]+)\s:\s", r"\1:", transcript)

    # Ensure lines with "\n" followed by ":" within 30 characters are normalized
    transcript = re.sub(r"([\w\s.,;&'-]{0,30})\n\s*:", r"\1:", transcript)

    return transcript


# Function to display the first 3 conversations from a transcript
def display_first_three_conversations(transcript):
    """Extract and display the first 3 speaker-dialogue pairs."""
    conversations = re.findall(
        r"([A-Za-zÀ-ÖØ-öø-ÿ.,;&'-]+(?: [A-Za-zÀ-ÖØ-öø-ÿ.,;&'-]+)*):\s*(.*?)(?=\n[A-Za-zÀ-ÖØ-öø-ÿ.,;&'-]+(?: [A-Za-zÀ-ÖØ-öø-ÿ.,;&'-]+)*:|\Z)",
        transcript,
        re.DOTALL,
    )
    print("First 3 Conversations:")
    for i, (speaker, content) in enumerate(conversations[:3], start=1):
        truncated_content = content.strip()[:2000]  # Limit content display
        print(
            f"{i}. {speaker}: {truncated_content}{'...' if len(content.strip()) > 2000 else ''}"
        )
    print("-" * 40)


# Function to group transcript by speaker
def group_transcript_by_speaker(transcript):
    """Organize transcript content by speaker."""
    speaker_dialogues = re.findall(
        r"([A-Za-zÀ-ÖØ-öø-ÿ.,;&'-]+(?: [A-Za-zÀ-ÖØ-öø-ÿ.,;&'-]+)*):\s*(.*?)(?=\n[A-Za-zÀ-ÖØ-öø-ÿ.,;&'-]+(?: [A-Za-zÀ-ÖØ-öø-ÿ.,;&'-]+)*:|\Z)",
        transcript,
        re.DOTALL,
    )
    grouped_content = {}
    speaker_original_names = {}
    for speaker, content in speaker_dialogues:
        normalized_name = normalize_speaker_name(speaker)
        if normalized_name not in grouped_content:
            grouped_content[normalized_name] = []
            speaker_original_names[normalized_name] = speaker
        grouped_content[normalized_name].append(content.strip())
    return grouped_content, speaker_original_names


# Function to parse user input for speaker selection
def parse_speaker_selection(input_str, num_speakers):
    """Parse user input allowing selection by number, range (1-3), or 'all'."""
    if input_str.strip().lower() == "all":
        return list(range(1, num_speakers + 1))
    selected_indices = set()
    for part in input_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            selected_indices.update(range(start, end + 1))
        elif part.isdigit():
            selected_indices.add(int(part))
    return sorted(i for i in selected_indices if 1 <= i <= num_speakers)


# Function to print selected speakers' content
def print_selected_speakers(grouped_content, speaker_original_names, selected_speakers):
    """Display and return content of selected speakers."""
    normalized_selected_speakers = {
        normalize_speaker_name(speaker) for speaker in selected_speakers
    }
    print("\nSelected Speakers' Content:")
    selected_data = {}
    for index, normalized_name in enumerate(normalized_selected_speakers, start=1):
        if normalized_name in grouped_content:
            original_name = speaker_original_names[normalized_name]
            selected_data[f"speaker{index}"] = {
                "name": original_name,
                "content": " ".join(grouped_content[normalized_name]),
            }
            print(
                f"Speaker: {original_name}\nContent:\n{' '.join(grouped_content[normalized_name])}\n{'-' * 40}"
            )
    return selected_data


# Main function to process the JSON
def process_transcripts():
    """Main script to process earnings call transcripts from a JSON file."""
    folder_path = input("Enter the folder path containing the JSON file: ").strip()
    json_file_path = os.path.join(folder_path, "earnings_transcripts_2020_2024.json")

    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: The file '{json_file_path}' does not exist.")
        return

    # Load JSON file
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Process each transcript in the JSON
    for index, entry in enumerate(data, start=1):
        print(
            f"Processing Entry {index} - {entry.get('symbol', f'Entry-{index}')} {entry.get('quarter', 'Q?')} {entry.get('year', 'Unknown Year')}"
        )
        transcript = entry.get("content", "")

        if transcript.strip():
            # Preprocess transcript
            transcript = preprocess_transcript(transcript)

            # Display first three conversations
            display_first_three_conversations(transcript)

            # Group transcript by speaker
            grouped_content, speaker_original_names = group_transcript_by_speaker(
                transcript
            )

            # Display available speakers
            unique_speakers = [
                speaker_original_names[name] for name in grouped_content.keys()
            ]
            print("Available speakers:")
            for i, speaker in enumerate(unique_speakers, start=1):
                print(f"{i}. {speaker}")

            # Ask user to select speakers
            selected_indices = parse_speaker_selection(
                input("Enter speaker numbers (e.g., 1,3-5,all): "), len(unique_speakers)
            )
            selected_speakers = [unique_speakers[i - 1] for i in selected_indices]

            # Save selected speakers' content
            selected_data = print_selected_speakers(
                grouped_content, speaker_original_names, selected_speakers
            )
            entry.update(selected_data)
        else:
            print(f"Skipping Entry {index}: No transcript available.")
        print("\n" + "=" * 60 + "\n")

    # Save updated JSON to a new file
    output_file = os.path.join(folder_path, "earnings_transcripts_2020_2024.json")
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"Processed JSON saved to {output_file}")


# Run the program
if __name__ == "__main__":
    process_transcripts()
