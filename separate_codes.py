import os
import re
import shutil

# Paths
input_folder = "ru-full/ru-txt/"
output_folder = "laws/ru/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def is_law_document(text, min_articles=1):
    # Look for "Article X." style patterns
    matches = re.findall(r'Статья+\s+\d+\.', text, flags=re.IGNORECASE)
    if len(matches) < min_articles:
        return False
    
    # Check if first few are sequential (1, 2, 3...)
    numbers = [int(re.search(r'\d+', m).group()) for m in matches]
    return all(numbers[i] == i + 1 for i in range(min(len(numbers), min_articles)))

# Process files
law_docs = 0
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if not os.path.isfile(file_path):
        continue

    if not filename.lower().endswith(".txt"):
        continue  # placeholder for other formats
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    if is_law_document(text):
        shutil.copy(file_path, os.path.join(output_folder, filename))
        print(f"Copied: {filename}")
        law_docs += 1


print("Files copied to:", output_folder)
print("Total files:", law_docs)
