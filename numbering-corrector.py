"""
Merged subarticle numbers correction (121-modda -> 12-1-modda)
"""

import os
import re

def correct_article_numbers_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return

    # Match patterns: "123-modda", "123 modda", "123 - modda", etc.
    pattern_uz = re.compile(r'^(\d+(?:-\d+)?)\s*-?\s*(modda)(\.?)', re.IGNORECASE | re.MULTILINE)
    matches = list(pattern_uz.finditer(original_content))

    if not matches:
        return

    corrected_content = original_content
    last_main_article_num = 0
    replacements = []

    for i in range(len(matches)):
        current_match = matches[i]
        current_article_str = current_match.group(1)

        if '-' in current_article_str:
            try:
                last_main_article_num = int(current_article_str.split('-')[0])
            except (ValueError, IndexError):
                pass
            continue

        current_article_num = int(current_article_str)

        # Check if this should be a hyphenated article
        if (0 < last_main_article_num < current_article_num and
                str(current_article_num).startswith(str(last_main_article_num)) and
                len(str(current_article_num)) == len(str(last_main_article_num)) + 1):

            original_full_match = current_match.group(0)
            prefix_len = len(str(last_main_article_num))
            corrected_number_str = f"{str(current_article_num)[:prefix_len]}-{str(current_article_num)[prefix_len:]}"
            corrected_full_match = f"{corrected_number_str}-modda{current_match.group(3)}"

            replacements.append({
                'start': current_match.start(),
                'end': current_match.end(),
                'original': original_full_match,
                'corrected': corrected_full_match
            })

            print(
                f"  - In {os.path.basename(filepath)}, correcting '{original_full_match.strip()}' to '{corrected_full_match.strip()}'")
        else:
            last_main_article_num = current_article_num

    # Apply replacements in reverse order to preserve positions
    if replacements:
        for replacement in reversed(replacements):
            corrected_content = (
                    corrected_content[:replacement['start']] +
                    replacement['corrected'] +
                    corrected_content[replacement['end']:]
            )

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(corrected_content)
            print(f"Successfully corrected and saved {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Error writing to file {filepath}: {e}")


def process_all_files(base_directory):
    uz_dir = os.path.join(base_directory, 'uz')

    if not os.path.isdir(uz_dir):
        print(f"Directory not found: {uz_dir}")
        return

    print(f"Processing files in: {uz_dir}")

    for filename in os.listdir(uz_dir):
        # Only process files that match the pattern: -<number>.txt
        if filename.startswith('-') and filename.endswith('.txt'):
            filepath = os.path.join(uz_dir, filename)
            correct_article_numbers_in_file(filepath)


if __name__ == "__main__":
    base_legal_docs_path = "./laws/"

    print("Starting Uzbek article number correction process.")
    print("=" * 40)

    process_all_files(base_legal_docs_path)

    print("\n" + "=" * 40)
    print("Processing complete.")