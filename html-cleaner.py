import os
import re

def clean_renamed_file(input_path, output_path):
    """
    Reads a text file and performs a multi-stage cleaning process to extract
    only the core readable text.
    """
    try:
        # Step 1: Read the file using 'utf-8-sig' to automatically handle the BOM.
        with open(input_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            content = f.read()

        # Step 2 (NEW): Remove entire <script> and <style> blocks.
        # This is the key step to remove CSS and JavaScript code.
        # The re.DOTALL flag allows '.' to match newlines, which is crucial for multi-line blocks.
        # The re.IGNORECASE flag handles tags like <SCRIPT> or <Style>.
        content = re.sub(r'<script.*?>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'<style.*?>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)

        # Step 3: Substitute the specific superscript numbering (e.g., 123<sup>4</sup> -> 123-4).
        cleaned_content = re.sub(r'(\d+)<sup>(\d+)</sup>', r'\1-\2', content)

        # Step 4: Remove all remaining HTML tags.
        text_without_html = re.sub(r'<.*?>', '', cleaned_content)

        # Step 5: Normalize whitespace and remove leftover junk.
        # This replaces all sequences of whitespace with a single space.
        normalized_text = ' '.join(text_without_html.split())

        # Step 6: Save the fully cleaned text to the output directory.
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(normalized_text)

        print(f"✅ Successfully cleaned {os.path.basename(input_path)} and saved to output.")

    except Exception as e:
        print(f"❌ An error occurred while cleaning {os.path.basename(input_path)}: {e}")


def process_directory_with_rename(input_dir, output_dir):
    """
    Pre-processes by renaming .doc files to .txt, then cleans them.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"\nScanning for .doc files in '{input_dir}' to rename and process...")

    # Loop through every file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.doc'):
            old_file_path = os.path.join(input_dir, filename)
            
            # Create the new filename by changing the extension to .txt
            base_filename = os.path.splitext(filename)[0]
            new_filename = base_filename + '.txt'
            renamed_file_path = os.path.join(input_dir, new_filename)

            # --- PRE-PROCESSING STEP ---
            try:
                print(f"Renaming {filename} to {new_filename}...")
                os.rename(old_file_path, renamed_file_path)
            except FileExistsError:
                print(f"⚠️  Warning: {new_filename} already exists. Skipping rename, will process existing file.")
            except Exception as e:
                print(f"❌ Could not rename {filename}: {e}")
                continue

            # --- PROCESSING THE RENAMED FILE ---
            output_file_path = os.path.join(output_dir, new_filename)
            clean_renamed_file(renamed_file_path, output_file_path)


if __name__ == "__main__":
    input_folder_uz = 'downloaded_docs/uz-doc'
    output_folder_uz = 'doc2txt/uz'

    input_folder_ru = 'downloaded_docs/ru-doc'
    output_folder_ru = 'doc2txt/ru'

    process_directory_with_rename(input_folder_ru, output_folder_ru)
    process_directory_with_rename(input_folder_uz, output_folder_uz)
