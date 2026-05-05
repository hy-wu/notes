import os
import json
import fitz
from datetime import datetime

def scan_folders():
    roots = ['C:/Users/hy-wu.DESKTOP-G355NC5/1', 'C:/Users/hy-wu.DESKTOP-G355NC5/Downloads']
    keywords = ['qgp', 'qcd', 'mc', 'monte', 'boltzmann', 'kinetic', 'hydro', 'fluid', 'spin', 'relativistic', 'bamps', 'particle', 'physics']
    output_path = 'C:/Users/hy-wu.DESKTOP-G355NC5/Documents/GitHub/notes/QGP_MC/QGP_Insight/papers_db.json'
    
    db = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            db = json.load(f)
    
    existing_paths = {p['path'] for p in db}
    
    for root in roots:
        if not os.path.exists(root): continue
        for r, d, files in os.walk(root):
            for f in files:
                if f.lower().endswith('.pdf'):
                    path = os.path.join(r, f).replace('\\', '/')
                    if path not in existing_paths and any(k in f.lower() for k in keywords):
                        db.append({
                            "filename": f,
                            "path": path,
                            "summary": None,
                            "tags": [k for k in keywords if k in f.lower()],
                            "scanned_at": datetime.now().isoformat()
                        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=4, ensure_ascii=False)
    print(f"Total papers in DB: {len(db)}")

if __name__ == "__main__":
    scan_folders()
