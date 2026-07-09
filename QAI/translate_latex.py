import os
import sys
import tarfile
import re
import json
import time
import hashlib
import random
import urllib.request
import urllib.parse
import ssl
import argparse

def get_paragraph_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def extract_tar_gz(tar_path, extract_dir):
    print(f"Extracting {tar_path} to {extract_dir}...")
    if not tarfile.is_tarfile(tar_path):
        print(f"Error: {tar_path} is not a valid tar archive (this paper might be a PDF-only submission on arXiv).")
        sys.exit(2)
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete.")

def call_gemini_api(api_key, text, model="gemini-1.5-pro"):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    prompt = f"You are a professional physics translator. Translate the following LaTeX text from English to Chinese. Keep all LaTeX command tags (e.g., \\cite{{...}}, \\ref{{...}}, \\label{{...}}, \\section{{...}}, \\emph{{...}}, \\textbf{{...}}) and math environments (e.g., $...$, $$...$$, \\begin{{equation}}...\\end{{equation}}) EXACTLY as they are. Do not translate variables, chemical symbols, or command names. Translate ONLY the actual English text inside or around them. Do not include any explanations or backticks, output ONLY the translated LaTeX code.\n\nLaTeX text:\n{text}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.1
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    with urllib.request.urlopen(req, context=ctx) as response:
        res = json.loads(response.read().decode('utf-8'))
        translated = res['candidates'][0]['content']['parts'][0]['text']
        return translated.strip()

def call_deepseek_api(api_key, text, model="deepseek-v4-flash"):
    url = "https://api.deepseek.com/chat/completions"
    
    system_prompt = (
        "You are a professional physics translator. Translate the following LaTeX text from English to Chinese. "
        "Keep all LaTeX command tags (e.g., \\cite{...}, \\ref{...}, \\label{...}, \\section{...}, \\emph{...}, \\textbf{...}) "
        "and math environments (e.g., $...$, $$...$$, \\begin{equation}...\\end{equation}) EXACTLY as they are. "
        "Do not translate variables or command names. Translate ONLY the actual English text inside or around them. "
        "Do not explain, output ONLY the translated LaTeX text."
    )
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
    )
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    with urllib.request.urlopen(req, context=ctx) as response:
        res = json.loads(response.read().decode('utf-8'))
        translated = res['choices'][0]['message']['content']
        return translated.strip()

def should_translate(paragraph):
    p = paragraph.strip()
    if not p:
        return False
    
    # Split into non-empty lines
    lines = [line.strip() for line in p.split('\n') if line.strip()]
    if not lines:
        return False
        
    # Don't translate pure comments (every line starts with %)
    if all(line.startswith('%') for line in lines):
        return False
        
    # Don't translate pure math environments / command definitions
    if p.startswith('\\def') or p.startswith('\\newcommand') or p.startswith('\\usepackage'):
        return False
    # Ignore document class, bibliography styles, etc.
    if p.startswith('\\documentclass') or p.startswith('\\bibliographystyle') or p.startswith('\\bibliography'):
        return False
    # Ignore pure figure inclusion if no captions
    if p.startswith('\\begin{figure}') and '\\caption' not in p:
        return False
    # Skip paragraphs consisting entirely of \input, \include, or \subfile commands
    if all(line.startswith('\\input') or line.startswith('\\include') or line.startswith('\\subfile') for line in lines):
        return False
    # Skip standard macro structures that don't have natural text
    if re.match(r'^\\[a-zA-Z]+\{[^}]+\}$', p) and not any(cmd in p for cmd in ['\\title', '\\section', '\\subsection', '\\subsubsection', '\\caption', '\\author']):
        return False
    return True

def translate_paragraph(api_type, api_key, model, paragraph, cache, cache_lock=None):
    p_hash = get_paragraph_hash(paragraph)
    
    if cache_lock:
        with cache_lock:
            if p_hash in cache:
                return cache[p_hash]
    else:
        if p_hash in cache:
            return cache[p_hash]
        
    if not should_translate(paragraph):
        return paragraph
        
    # Strip leading/trailing environment tags to prevent LLM from duplicating or auto-closing them
    prefix = ""
    suffix = ""
    p_stripped = paragraph.strip()
    
    # Match leading environment begins (e.g. \begin{abstract}, \begin{equation})
    begin_match = re.match(r'^(\\begin\{[a-zA-Z*]+\}(?:\[[^\]]*\])?(?:\{[^}]*\})?\s*)', p_stripped)
    if begin_match:
        prefix = begin_match.group(1)
        p_stripped = p_stripped[len(prefix):]
        
    # Match trailing environment ends (e.g. \end{abstract})
    end_match = re.search(r'(\s*\\end\{[a-zA-Z*]+\})$', p_stripped)
    if end_match:
        suffix = end_match.group(1)
        p_stripped = p_stripped[:-len(suffix)]
        
    if not p_stripped.strip():
        # If paragraph is only environment tags (e.g. \begin{keywords}\n\end{keywords})
        res = ""
    else:
        # Stagger concurrent requests slightly
        if cache_lock:
            time.sleep(random.uniform(0.0, 0.5))
        print(f"Translating paragraph ({len(p_stripped)} chars)...")
        # Retry logic
        for attempt in range(5):
            try:
                if api_type == 'gemini':
                    res = call_gemini_api(api_key, p_stripped, model)
                else:
                    res = call_deepseek_api(api_key, p_stripped, model)
                break
            except Exception as e:
                print(f"  API call failed (attempt {attempt+1}/5): {e}")
                time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
        else:
            raise Exception("API translation failed after 5 attempts.")
            
    final_res = prefix + res + suffix
    
    if cache_lock:
        with cache_lock:
            cache[p_hash] = final_res
    else:
        cache[p_hash] = final_res
        
    return final_res

def rewrite_input_paths(content, base_dir):
    # Match \input{path} or \include{path} or \subfile{path}
    def repl(match):
        cmd = match.group(1)
        path = match.group(2).strip()
        
        clean_path = path[:-4] if path.endswith('.tex') else path
        if clean_path.endswith('_ZH'):
            return match.group(0)
            
        possible_paths = [clean_path + '.tex', clean_path]
        exists = False
        for p in possible_paths:
            full_p = os.path.normpath(os.path.join(base_dir, p))
            if os.path.exists(full_p):
                exists = True
                break
                
        if exists:
            if '/' in clean_path:
                dir_part, file_part = clean_path.rsplit('/', 1)
                new_path = f"{dir_part}/{file_part}_ZH"
            elif '\\' in clean_path:
                dir_part, file_part = clean_path.rsplit('\\', 1)
                new_path = f"{dir_part}\\{file_part}_ZH"
            else:
                new_path = f"{clean_path}_ZH"
            return f"{cmd}{{{new_path}}}"
        else:
            return match.group(0)

    content = re.sub(r'(\\input|\\include|\\subfile)\{([^}]+)\}', repl, content)
    return content

def translate_tex_file(api_type, api_key, model, file_path, output_path, cache_path):
    print(f"Translating: {file_path} -> {output_path}")
    
    # Load cache
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            pass

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Upgrade LaTeX 2.09 \documentstyle to LaTeX2e \documentclass
    if '\\documentstyle' in content:
        print("  Detected LaTeX 2.09 document. Upgrading to LaTeX2e...")
        content = re.sub(
            r'\\documentstyle\[([^\]]*)\]\{revtex\}',
            r'\\documentclass[\1,twocolumn]{revtex4-2}\n\\usepackage{graphicx}\n\\providecommand{\\draft}{}\n\\providecommand{\\address}[1]{\\affiliation{#1}}\n\\newdimen\\epsfxsize\n\\providecommand{\\epsffile}[2][]{\\includegraphics[width=\\epsfxsize]{#2}}\n',
            content
        )
        content = re.sub(
            r'\\documentstyle\{revtex\}',
            r'\\documentclass[twocolumn]{revtex4-2}\n\\usepackage{graphicx}\n\\providecommand{\\draft}{}\n\\providecommand{\\address}[1]{\\affiliation{#1}}\n\\newdimen\\epsfxsize\n\\providecommand{\\epsffile}[2][]{\\includegraphics[width=\\epsfxsize]{#2}}\n',
            content
        )
        
        # Remove old \twocolumn[...] title page wrapper
        content = re.sub(
            r'\\twocolumn\[\s*\\hsize\s*\\textwidth\s*\\columnwidth\s*\\hsize\s*\\csname\s*@twocolumnfalse\\endcsname',
            '',
            content
        )
        content = re.sub(
            r'\\twocolumn\[[^\]]*@twocolumnfalse\\endcsname',
            '',
            content
        )
        
        # Strip the closing bracket ']' of the \twocolumn block.
        content = re.sub(r'(\\pacs\{[^\}]+\})\s*\]', r'\1', content)
        content = re.sub(r'(\\end\{abstract\})\s*\]', r'\1', content)
        content = re.sub(r'\n\s*\]\s*\n(?=\s*\\section)', r'\n', content)

    # Disable pdfcprot package if present (incompatible with XeLaTeX)
    content = re.sub(r'\\usepackage\[[^\]]*\]\{pdfcprot\}', '% \\\\usepackage{pdfcprot}', content)
    content = re.sub(r'\\usepackage\{pdfcprot\}', '% \\\\usepackage{pdfcprot}', content)

    # General pdfximage -> XeTeX pdfpagecount replacement
    if '\\pdfximage' in content:
        print("  Detected \\pdfximage usage. Injecting XeTeX compatibility patch...")
        def pdfximage_repl(match):
            filename = match.group(1).strip()
            def_or_edef = match.group(2).strip()
            macro = match.group(3).strip()
            macro_name = macro[1:] # strip leading backslash
            return (
                f"\\newcount\\{macro_name}count\n"
                f"\\ifx\\XeTeXpdfpagecount\\undefined\n"
                f"  \\pdfximage{{{filename}}}\n"
                f"  \\{macro_name}count=\\pdflastximagepages\n"
                f"\\else\n"
                f"  \\{macro_name}count=\\XeTeXpdfpagecount\"{filename}\"\n"
                f"\\fi\n"
                f"\\edef{macro}{{\\the\\{macro_name}count}}\n"
            )
        content = re.sub(
            r'\\pdfximage\s*\{([^}]+)\}\s*\\(def|edef)\s*(\\[a-zA-Z@]+)\s*\{\s*\\the\s*\\pdflastximagepages\s*\}',
            pdfximage_repl,
            content
        )

    # Inhibit translation of preamble, inject ctex package
    preamble = ""
    body = content
    if '\\begin{document}' in content:
        parts = content.split('\\begin{document}', 1)
        preamble = parts[0]
        body = parts[1]
        
        # Upgrade revtex4-1 to revtex4-2
        preamble = preamble.replace('revtex4-1', 'revtex4-2')
        
        # Inject switch@array and math symbol undefine patches after documentclass
        if '\\switch@array' not in preamble:
            doc_match = re.search(r'\\documentclass\[[^\]]*\]\{[^}]+\}', preamble)
            if not doc_match:
                doc_match = re.search(r'\\documentclass\{[^}]+\}', preamble)
            if doc_match:
                end_pos = doc_match.end()
                patch = (
                    "\n\\makeatletter\n"
                    "\\def\\switch@array{}\n"
                    "% Save and undefine math symbols to prevent clash with fontspec/ctex\n"
                    "\\let\\origGamma\\Gamma \\let\\Gamma\\undefined\n"
                    "\\let\\origDelta\\Delta \\let\\Delta\\undefined\n"
                    "\\let\\origTheta\\Theta \\let\\Theta\\undefined\n"
                    "\\let\\origLambda\\Lambda \\let\\Lambda\\undefined\n"
                    "\\let\\origXi\\Xi \\let\\Xi\\undefined\n"
                    "\\let\\origPi\\Pi \\let\\Pi\\undefined\n"
                    "\\let\\origSigma\\Sigma \\let\\Sigma\\undefined\n"
                    "\\let\\origUpsilon\\Upsilon \\let\\Upsilon\\undefined\n"
                    "\\let\\origPhi\\Phi \\let\\Phi\\undefined\n"
                    "\\let\\origPsi\\Psi \\let\\Psi\\undefined\n"
                    "\\let\\origOmega\\Omega \\let\\Omega\\undefined\n"
                    "\\makeatother\n"
                )
                preamble = preamble[:end_pos] + patch + preamble[end_pos:]

        # Inject ctex into preamble
        if '\\usepackage[UTF8]{ctex}' not in preamble:
            preamble += "\n\\usepackage[UTF8]{ctex}\n"
            
        pass
    else:
        # If no \begin{document}, treat the whole file as body (sub-file)
        pass

    # Split body into paragraphs
    paragraphs = body.split('\n\n')
    translated_paragraphs = [None] * len(paragraphs)
    
    # Identify which paragraphs need translation
    to_translate = {}
    in_bibliography = False
    for i, p in enumerate(paragraphs):
        if '\\begin{thebibliography}' in p:
            in_bibliography = True
            
        p_hash = get_paragraph_hash(p)
        if in_bibliography or not should_translate(p):
            translated_paragraphs[i] = p
        elif p_hash in cache:
            translated_paragraphs[i] = cache[p_hash]
        else:
            to_translate[i] = p
            
    if to_translate:
        print(f"Translating {len(to_translate)} paragraphs concurrently...")
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        cache_lock = threading.Lock()
        max_workers = 15
        
        def worker(idx, p):
            trans_p = translate_paragraph(api_type, api_key, model, p, cache, cache_lock)
            return idx, trans_p
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(worker, idx, p): idx for idx, p in to_translate.items()}
            
            last_save = time.time()
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    index, trans_p = future.result()
                    translated_paragraphs[index] = trans_p
                except Exception as e:
                    print(f"Error translating paragraph {idx+1}: {e}")
                    translated_paragraphs[idx] = to_translate[idx]
                    
                with cache_lock:
                    if time.time() - last_save > 10:
                        with open(cache_path, 'w', encoding='utf-8') as f:
                            json.dump(cache, f, ensure_ascii=False, indent=2)
                        last_save = time.time()
    
    # Fallback/Safety check: Ensure no None in translated_paragraphs
    for i in range(len(translated_paragraphs)):
        if translated_paragraphs[i] is None:
            translated_paragraphs[i] = paragraphs[i]

    # Save final cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Reassemble file
    translated_body = '\n\n'.join(translated_paragraphs)
    
    # Restore math symbols right after \begin{document}
    restore_patch = (
        "\n\\makeatletter\n"
        "\\ifx\\Gamma\\undefined \\let\\Gamma\\origGamma \\fi\n"
        "\\ifx\\Delta\\undefined \\let\\Delta\\origDelta \\fi\n"
        "\\ifx\\Theta\\undefined \\let\\Theta\\origTheta \\fi\n"
        "\\ifx\\Lambda\\undefined \\let\\Lambda\\origLambda \\fi\n"
        "\\ifx\\Xi\\undefined \\let\\Xi\\origXi \\fi\n"
        "\\ifx\\Pi\\undefined \\let\\Pi\\origPi \\fi\n"
        "\\ifx\\Sigma\\undefined \\let\\Sigma\\origSigma \\fi\n"
        "\\ifx\\Upsilon\\undefined \\let\\Upsilon\\origUpsilon \\fi\n"
        "\\ifx\\Phi\\undefined \\let\\Phi\\origPhi \\fi\n"
        "\\ifx\\Psi\\undefined \\let\\Psi\\origPsi \\fi\n"
        "\\ifx\\Omega\\undefined \\let\\Omega\\origOmega \\fi\n"
        "\\makeatother\n"
    )
    
    if '\\begin{document}' in content:
        final_content = preamble + '\\begin{document}' + restore_patch + translated_body
    else:
        final_content = translated_body

    # Rewrite input paths to target translated subfiles
    final_content = rewrite_input_paths(final_content, os.path.dirname(file_path))

    # Ensure space between command and non-ASCII character (e.g. \it友好 -> \it 友好)
    # to prevent XeTeX from parsing them as a single macro name
    final_content = re.sub(r'(\\[a-zA-Z]+)([^\x00-\x7F])', r'\1 \2', final_content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print(f"Translation saved to {output_path}")

def compile_pdf(tex_dir, tex_filename):
    print(f"Compiling {tex_filename} in {tex_dir} using xelatex...")
    orig_cwd = os.getcwd()
    base_name = tex_filename[:-4] if tex_filename.endswith('.tex') else tex_filename
    null_device = 'NUL' if os.name == 'nt' else '/dev/null'
    try:
        os.chdir(tex_dir)
        # 1. Run xelatex once to generate aux
        print("Running xelatex (1/3)...")
        os.system(f"xelatex -interaction=nonstopmode {tex_filename} < {null_device}")
        
        # 2. Run bibtex to build bibliography
        print("Running bibtex...")
        os.system(f"bibtex {base_name} < {null_device}")
        
        # Fallback: if bibtex failed or did not resolve, try copying legacy .bbl
        orig_bbl = "main.bbl"
        if not os.path.exists(orig_bbl):
            bbl_files = [f for f in os.listdir('.') if f.endswith('.bbl') and f != f"{base_name}.bbl"]
            if bbl_files:
                orig_bbl = bbl_files[0]
        
        target_bbl = f"{base_name}.bbl"
        if os.path.exists(orig_bbl):
            print(f"Copying {orig_bbl} to {target_bbl} as bibliography fallback...")
            import shutil
            shutil.copy(orig_bbl, target_bbl)
            
        # 3. Run xelatex twice more to resolve citations and cross-references
        for i in range(2):
            print(f"Running xelatex ({i+2}/3)...")
            os.system(f"xelatex -interaction=nonstopmode {tex_filename} < {null_device}")
            
        print("Compilation complete.")
    except Exception as e:
        print(f"Compilation error: {e}")
    finally:
        os.chdir(orig_cwd)

def load_env():
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

def main():
    load_env()
    parser = argparse.ArgumentParser(description="Auto-translate and compile LaTeX documents using LLMs.")
    parser.add_argument("--file", help="Path to RMP arXiv LaTeX package (.tar.gz)")
    parser.add_argument("--dir", help="Path to extracted LaTeX source directory")
    parser.add_argument("--api", choices=["gemini", "deepseek"], default="gemini", help="API type to use")
    parser.add_argument("--key", help="API Key (falls back to GEMINI_API_KEY or DEEPSEEK_API_KEY env vars)")
    parser.add_argument("--model", help="Specific LLM model to use")
    
    args = parser.parse_args()
    
    api_key = args.key
    if not api_key:
        if args.api == 'gemini':
            api_key = os.environ.get("GEMINI_API_KEY")
        else:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            
    if not api_key:
        print(f"Error: API Key must be provided via --key or set as {args.api.upper()}_API_KEY environment variable.")
        sys.exit(1)
        
    model = args.model
    if not model:
        model = "gemini-3.5-flash" if args.api == 'gemini' else "deepseek-v4-flash"

    tar_path = args.file
    extract_dir = None
    workspace_dir = None
    
    if args.dir:
        extract_dir = os.path.abspath(args.dir)
        workspace_dir = os.path.dirname(extract_dir)
        print(f"Using direct LaTeX source directory: {extract_dir}")
    elif tar_path:
        if not os.path.exists(tar_path):
            print(f"Error: source archive not found at {tar_path}")
            sys.exit(1)
            
        filename = os.path.basename(tar_path)
        folder_name = filename.replace('.tar.gz', '').replace('.gz', '')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        livf_dir = os.path.dirname(script_dir)
        workspace_dir = os.path.join(livf_dir, "papers", "translated", folder_name)
        os.makedirs(workspace_dir, exist_ok=True)
        
        extract_dir = os.path.join(workspace_dir, "src")
        extract_tar_gz(tar_path, extract_dir)
    else:
        print("Error: Either --file or --dir must be specified.")
        sys.exit(1)
    
    tex_files = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith('.tex') and not f.endswith('_ZH.tex'):
                rel_path = os.path.relpath(os.path.join(root, f), extract_dir)
                tex_files.append(rel_path)
                
    if not tex_files:
        print("Error: No .tex files found in the extracted package.")
        sys.exit(1)
        
    # Sort tex files by file size in descending order to prefer the main manuscript
    tex_files.sort(key=lambda f: os.path.getsize(os.path.join(extract_dir, f)), reverse=True)
    print(f"Found .tex files to translate (sorted by size): {tex_files}")
    
    main_file = None
    # Prioritize root-level .tex files that contain \begin{document}
    for f in tex_files:
        if '/' not in f and '\\' not in f:
            with open(os.path.join(extract_dir, f), 'r', encoding='utf-8', errors='ignore') as tf:
                if '\\begin{document}' in tf.read():
                    main_file = f
                    break
                    
    # Fallback to any .tex file containing \begin{document}
    if not main_file:
        for f in tex_files:
            with open(os.path.join(extract_dir, f), 'r', encoding='utf-8', errors='ignore') as tf:
                if '\\begin{document}' in tf.read():
                    main_file = f
                    break
                
    if not main_file:
        main_file = tex_files[0]
        print(f"Warning: Could not determine main file. Defaulting to {main_file}")
    else:
        print(f"Detected main document file: {main_file}")

    for f in tex_files:
        src_path = os.path.join(extract_dir, f)
        base_dir = os.path.dirname(src_path)
        base_name = os.path.basename(f)[:-4]
        out_name = f"{base_name}_ZH.tex"
        out_path = os.path.join(base_dir, out_name)
        
        # Replace '/' or '\' in relpath to keep cache files in the workspace root
        safe_cache_name = f.replace('/', '_').replace('\\', '_')[:-4]
        cache_path = os.path.join(workspace_dir, f"{safe_cache_name}_translate_cache.json")
        
        translate_tex_file(args.api, api_key, model, src_path, out_path, cache_path)
        
    main_base = main_file[:-4]
    compile_pdf(extract_dir, f"{main_base}_ZH.tex")
    
    final_pdf = os.path.join(extract_dir, f"{main_base}_ZH.pdf")
    if os.path.exists(final_pdf):
        print(f"\nSUCCESS! Translated PDF is available at: {final_pdf}")
    else:
        print("\nCompilation failed to produce PDF. Check the logs in the src folder.")

if __name__ == "__main__":
    main()
