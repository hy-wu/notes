import os
import sys
import subprocess
import json
import time

def batch_translate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    livf_dir = os.path.dirname(script_dir)
    
    # Resolve source directory dynamically (Google Drive on macOS/Windows, or fallback in repo)
    mac_path = "/Users/bjergsen/Library/CloudStorage/GoogleDrive-wwwgreta20210915@gmail.com/我的云端硬盘/ReviewsModernPhysics/RMP_LaTeX_Sources"
    win_roots = [
        r"G:\我的云端硬盘\ReviewsModernPhysics\RMP_LaTeX_Sources",
        r"G:\My Drive\ReviewsModernPhysics\RMP_LaTeX_Sources",
        r"H:\我的云端硬盘\ReviewsModernPhysics\RMP_LaTeX_Sources",
        r"H:\My Drive\ReviewsModernPhysics\RMP_LaTeX_Sources",
        os.path.expanduser(r"~\Google Drive\ReviewsModernPhysics\RMP_LaTeX_Sources")
    ]
    
    source_dir = mac_path
    if not os.path.exists(source_dir):
        for path in win_roots:
            if os.path.exists(path):
                source_dir = path
                break
        else:
            source_dir = os.path.join(livf_dir, "papers", "sources")
            
    script_path = os.path.join(script_dir, "translate_latex.py")
    log_path = os.path.join(script_dir, "batch_translation_log.json")
    
    # 1. Read API key from ~/.env
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
                    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found at {source_dir}")
        return

    # Find all RMP LaTeX archives
    archives = sorted([f for f in os.listdir(source_dir) if f.endswith('.tar.gz') or f.endswith('.gz')])
    print(f"Found {len(archives)} LaTeX source packages to translate.")

    # Load log if it exists
    status_log = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                status_log = json.load(f)
        except Exception:
            pass

    for i, archive in enumerate(archives, 1):
        print(f"\n========================================")
        print(f"[{i}/{len(archives)}] Processing: {archive}")
        print(f"========================================")
        
        archive_path = os.path.join(source_dir, archive)
        
        # Check if already successfully translated in log
        if status_log.get(archive, {}).get("status") == "SUCCESS":
            # Double check if translated PDF actually exists
            folder_name = archive.replace('.tar.gz', '').replace('.gz', '')
            workspace_dir = os.path.join(livf_dir, "papers", "translated", folder_name)
            # Scan for *_ZH.pdf in workspace_dir/src
            src_dir = os.path.join(workspace_dir, "src")
            pdf_exists = False
            if os.path.exists(src_dir):
                pdf_exists = any(f.endswith('_ZH.pdf') for f in os.listdir(src_dir))
                
            if pdf_exists:
                print(f"  Skipped (already successfully translated and compiled).")
                continue
        
        status_log[archive] = {
            "status": "RUNNING",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(status_log, f, indent=2)

        # Run translation script
        cmd = [
            sys.executable, script_path,
            "--file", archive_path,
            "--api", "deepseek",
            "--key", api_key,
            "--model", "deepseek-v4-flash"
        ]
        
        try:
            p = subprocess.Popen(cmd)
            p.communicate() # Wait for completion of this paper
            
            if p.returncode == 2:
                print(f"  SKIPPED: PDF-only submission on arXiv, no LaTeX sources.")
                status_log[archive] = {
                    "status": "SKIPPED_PDF_ONLY",
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                # Check if translation output exists
                folder_name = archive.replace('.tar.gz', '').replace('.gz', '')
                workspace_dir = os.path.join(livf_dir, "papers", "translated", folder_name)
                src_dir = os.path.join(workspace_dir, "src")
                
                pdf_file = None
                if os.path.exists(src_dir):
                    pdfs = [f for f in os.listdir(src_dir) if f.endswith('_ZH.pdf')]
                    if pdfs:
                        pdf_file = pdfs[0]
                
                if pdf_file:
                    print(f"  SUCCESS! Created PDF: {pdf_file}")
                    status_log[archive] = {
                        "status": "SUCCESS",
                        "pdf": pdf_file,
                        "end_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    print(f"  FAILED: Compilation did not produce a translated PDF.")
                    status_log[archive] = {
                        "status": "FAILED_COMPILATION",
                        "end_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
        except Exception as e:
            print(f"  ERROR during execution: {e}")
            status_log[archive] = {
                "status": f"ERROR: {e}",
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(status_log, f, indent=2)
            
        # Wait between books to cooling down
        # time.sleep(5)

if __name__ == "__main__":
    batch_translate()
