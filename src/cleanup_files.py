import os

ROOT_DIR = "../data/data_image"

removed_count = 0

for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.startswith("."):
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

print(f"Removed {removed_count} hidden files.")
