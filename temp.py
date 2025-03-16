import os

# Name of the file to delete
target_file = "CAR_after_earnings_updated.csv"

# Walk through all subdirectories in the current directory
for root, dirs, files in os.walk("."):
    if target_file in files:
        file_path = os.path.join(root, target_file)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
