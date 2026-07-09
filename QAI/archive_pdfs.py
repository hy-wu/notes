import os
import shutil

# Target directories relative to the current directory (QAI)
src_dir = "."
dest_dir = "PDFs"

os.makedirs(dest_dir, exist_ok=True)

# Iterate through subdirectories in QAI
count = 0
for item in os.listdir(src_dir):
    item_path = os.path.join(src_dir, item)
    
    # Exclude PDFs directory itself
    if item == "PDFs":
        continue
        
    if os.path.isdir(item_path):
        for file_name in os.listdir(item_path):
            if file_name.endswith(".pdf"):
                src_file = os.path.join(item_path, file_name)
                # Construct target file name: <subdir_name>_<file_name>
                dest_file_name = f"{item}_{file_name}"
                dest_file = os.path.join(dest_dir, dest_file_name)
                
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {item}/{file_name} -> PDFs/{dest_file_name}")
                count += 1
        # latex_path = os.path.join(item_path, "latex_source")
        # if os.path.isdir(latex_path):
        #     # Check for any PDF ending with _ZH.pdf inside latex_source
        #     for file_name in os.listdir(latex_path):
        #         if file_name.endswith("_ZH.pdf"):
        #             src_file = os.path.join(latex_path, file_name)
        #             # Construct target file name: <subdir_name>_<file_name>
        #             dest_file_name = f"{item}_{file_name}"
        #             dest_file = os.path.join(dest_dir, dest_file_name)
                    
        #             shutil.copy2(src_file, dest_file)
        #             print(f"Copied: {item}/latex_source/{file_name} -> PDFs/{dest_file_name}")
        #             count += 1

print(f"Finished. Total copied: {count} files.")
