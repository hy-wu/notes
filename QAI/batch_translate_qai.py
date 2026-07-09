import os
import subprocess
import sys
import json
import time

# Dynamic paths resolved relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIVF_DIR = os.path.dirname(SCRIPT_DIR)

# QAI directory under notes repository
QAI_DIR = "c:/Users/hy-wu.DESKTOP-G355NC5/Documents/GitHub/notes/QAI"
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "translate_latex.py")
# Load env from ~/Documents/Github/.env
env_path = os.path.expanduser("~/Documents/Github/.env")
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip("'").strip('"')

API_KEY = os.environ.get("DEEPSEEK_API_KEY")

def get_qai_papers():
    papers = []
    if not os.path.exists(QAI_DIR):
        print(f"Error: QAI directory not found at {QAI_DIR}")
        return papers
        
    for item in sorted(os.listdir(QAI_DIR)):
        item_path = os.path.join(QAI_DIR, item)
        if not os.path.isdir(item_path):
            continue
            
        latex_source_dir = os.path.join(item_path, "latex_source")
        if os.path.exists(latex_source_dir):
            # Check if there are any tex files to translate
            tex_files = [f for f in os.listdir(latex_source_dir) if f.endswith('.tex') and not f.endswith('_ZH.tex')]
            if tex_files:
                papers.append({
                    'folder_name': item,
                    'dir_path': latex_source_dir,
                    'tex_files': tex_files
                })
    return papers

def batch_translate_qai():
    papers = get_qai_papers()
    print(f"Found {len(papers)} papers with LaTeX sources in QAI.")
    
    for idx, paper in enumerate(papers, 1):
        folder = paper['folder_name']
        latex_dir = paper['dir_path']
        
        # Check if already translated (translated PDF exists)
        pdf_exists = any(f.endswith('_ZH.pdf') for f in os.listdir(latex_dir))
        if pdf_exists:
            print(f"[{idx}/{len(papers)}] Skipping {folder} (Already translated and compiled).")
            continue
            
        print(f"\n========================================")
        print(f"[{idx}/{len(papers)}] Translating: {folder}")
        print(f"========================================")
        
        cmd = [
            sys.executable, SCRIPT_PATH,
            "--dir", latex_dir,
            "--api", "deepseek",
            "--key", API_KEY,
            "--model", "deepseek-v4-flash"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            p = subprocess.Popen(cmd)
            p.communicate()
            if p.returncode == 0:
                print(f"Successfully translated {folder}")
            else:
                print(f"Error or compilation warning for {folder} (Exit code: {p.returncode})")
        except Exception as e:
            print(f"Exception during translation of {folder}: {e}")
            
        # Sleep to cool down and respect API rate limits
        time.sleep(2)

if __name__ == "__main__":
    batch_translate_qai()
