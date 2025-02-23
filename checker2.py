import os
import json


def check_speaker1_in_folders(base_path):
    total_files = 0
    files_with_issues = 0
    files_without_issues = 0
    folders_with_issues = []

    # Traverse all folders from the base path
    for root, dirs, files in os.walk(base_path):
        if "earnings_transcripts_2020_2024.json" in files:
            total_files += 1
            file_path = os.path.join(root, "earnings_transcripts_2020_2024.json")

            try:
                with open(file_path, "r") as file:
                    data = json.load(file)  # Load JSON data from file

                has_issues = False
                for entry in data:
                    if "speaker1" not in entry or entry["speaker1"] == "":
                        has_issues = True
                        break  # No need to check further, folder has issues

                if has_issues:
                    files_with_issues += 1
                    folders_with_issues.append(root)
                else:
                    files_without_issues += 1

            except Exception:
                # If there's an error reading the file, it won't be counted
                pass

    # Print results
    for folder in folders_with_issues:
        print(folder)
    print(f"Total files found: {total_files}")
    print(f"Files with 'speaker1' issues: {files_with_issues}")
    print(f"Files without 'speaker1' issues: {files_without_issues}")


# Start from the current directory
base_path = os.getcwd()
check_speaker1_in_folders(base_path)
