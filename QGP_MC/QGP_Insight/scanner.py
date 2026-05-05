import os
import json
import fitz  # PyMuPDF
from datetime import datetime

def scan_folders():
    roots = ['C:/Users/hy-wu.DESKTOP-G355NC5/1', 'C:/Users/hy-wu.DESKTOP-G355NC5/Downloads']
    keywords = ['qgp', 'qcd', 'mc', 'monte', 'boltzmann', 'kinetic', 'hydro', 'fluid', 'spin', 'relativistic', 'bamps', 'particle', 'physics']
    
    db = []
    
    for root in roots:
        if not os.path.exists(root): continue
        for r, d, files in os.walk(root):
            for f in files:
                if f.lower().endswith('.pdf'):
                    path = os.path.join(r, f).replace('\\', '/')
                    # Basic match
                    if any(k in f.lower() for k in keywords):
                        db.append({
                            "filename": f,
                            "path": path,
                            "summary": None,
                            "tags": [k for k in keywords if k in f.lower()],
                            "scanned_at": datetime.now().isoformat()
                        })
    
    with open('papers_db.json', 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=4, ensure_ascii=False)
    print(f"Scanned {len(db)} papers.")

if __name__ == "__main__":
    scan_folders()
