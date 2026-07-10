import os
import shutil
import json

# Target directories relative to the current directory (QAI)
src_dir = "."
dest_dir = "PDFs"

# Re-create PDFs directory to clean up previous archive files
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
os.makedirs(dest_dir, exist_ok=True)

# Iterate through subdirectories in QAI
count = 0
for item in os.listdir(src_dir):
    item_path = os.path.join(src_dir, item)
    
    # Exclude PDFs directory itself
    if item == "PDFs":
        continue
        
    if os.path.isdir(item_path):
        # Try to load arxiv_id and pdf_path from metadata.json
        arxiv_id = ""
        specified_pdf = ""
        metadata_path = os.path.join(item_path, "metadata.json")
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    val_arxiv = meta.get("arxiv_id")
                    arxiv_id = val_arxiv.strip() if isinstance(val_arxiv, str) else ""
                    val_pdf = meta.get("pdf_path")
                    specified_pdf = val_pdf.strip() if isinstance(val_pdf, str) else ""
            except Exception as e:
                print(f"Warning: failed to read metadata for {item}: {e}")
        
        # Build file prefix
        prefix = f"{arxiv_id}_" if arxiv_id else ""
        
        # 1. Archive English PDF
        en_copied = False
        if specified_pdf:
            src_file = os.path.join(item_path, specified_pdf)
            if os.path.isfile(src_file):
                dest_file_name = f"{prefix}{item}.pdf"
                dest_file = os.path.join(dest_dir, dest_file_name)
                shutil.copy2(src_file, dest_file)
                print(f"Copied EN: {item}/{specified_pdf} -> PDFs/{dest_file_name}")
                en_copied = True
                count += 1
                
        # If not specified or file not found, fallback to searching for any .pdf directly in item_path
        if not en_copied:
            for file_name in os.listdir(item_path):
                if file_name.endswith(".pdf") and not os.path.isdir(os.path.join(item_path, file_name)):
                    src_file = os.path.join(item_path, file_name)
                    dest_file_name = f"{prefix}{item}.pdf"
                    dest_file = os.path.join(dest_dir, dest_file_name)
                    shutil.copy2(src_file, dest_file)
                    print(f"Copied EN (fallback): {item}/{file_name} -> PDFs/{dest_file_name}")
                    en_copied = True
                    count += 1
                    break  # only copy one English pdf per folder
                    
        # 2. Archive Chinese PDF from latex_source
        latex_path = os.path.join(item_path, "latex_source")
        if os.path.isdir(latex_path):
            for file_name in os.listdir(latex_path):
                if file_name.endswith("_ZH.pdf"):
                    src_file = os.path.join(latex_path, file_name)
                    dest_file_name = f"{prefix}{item}_ZH.pdf"
                    dest_file = os.path.join(dest_dir, dest_file_name)
                    shutil.copy2(src_file, dest_file)
                    print(f"Copied ZH: {item}/latex_source/{file_name} -> PDFs/{dest_file_name}")
                    count += 1

print(f"\nFinished. Total copied: {count} files.")

