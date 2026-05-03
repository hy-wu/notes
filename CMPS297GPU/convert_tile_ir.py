import os
import re
import sys

# Windows
if os.name == 'nt':
    files_dir = r"C:\Users\hy-wu.DESKTOP-G355NC5\1\note\tile"
# WSL Ubuntu
elif os.name == 'posix':
    files_dir = r"/mnt/c/Users/hy-wu.DESKTOP-G355NC5/1/note/tile"
else:
    print("Unsupported OS")
    sys.exit(1)
copied_images_dir = "images"

original_files = [
    "1. Introduction — Tile IR.md",
    "2. Programming Model — Tile IR.md",
    "3. Syntax — Tile IR.md",
    "4. Binary Format — Tile IR.md",
    "5. Type System — Tile IR.md",
    "6. Semantics — Tile IR.md",
    "7. Memory Model — Tile IR.md",
    "8. Operations — Tile IR.md",
    "9. Debug Info — Tile IR.md"
]

translated_files = [
    "ds_translated_1. Introduction — Tile IR.md",
    "ds_translated_2. Programming Model — Tile IR.md",
    "ds_translated_3. Syntax — Tile IR.md",
    "ds_translated_4. Binary Format — Tile IR.md",
    "ds_translated_5. Type System — Tile IR.md",
    "ds_translated_6. Semantics — Tile IR.md",
    "ds_translated_7. Memory Model — Tile IR.md",
    "ds_translated_8. Operations — Tile IR.md",
    "ds_translated_9. Debug Info — Tile IR.md"
]

def escape_latex(text, in_math=False):
    if in_math:
        return text
    # Fix common Unicode characters that break LaTeX
    text = text.replace('−', '-') # U+2212 Minus
    text = text.replace('—', '---') # Em dash
    text = text.replace('–', '--')  # En dash
    text = text.replace('“', "``")
    text = text.replace('”', "''")
    text = text.replace('‘', "`")
    text = text.replace('’', "'")
    
    specials = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    res = ""
    i = 0
    while i < len(text):
        if text[i] in specials:
            res += specials[text[i]]
        elif text[i] == '\\':
            res += r'\textbackslash{}'
        else:
            res += text[i]
        i += 1
    return res

def process_inline(text):
    placeholders = []
    def add_placeholder(s):
        placeholders.append(s)
        return f"__PLACEHOLDER_{len(placeholders)-1}__"

    # 1. Math - Handle $$...$$ first, then $...$
    # We use a non-greedy match that covers potential single-line display math
    text = re.sub(r'\$\$.*?\$\$', lambda m: add_placeholder(m.group(0)), text)
    text = re.sub(r'\$.*?\$', lambda m: add_placeholder(m.group(0)), text)
    
    # 2. Code
    text = re.sub(r'`([^`]+)`', lambda m: add_placeholder(r'\texttt{' + escape_latex(m.group(1)) + '}'), text)

    # 3. Links [text](url) -> \href{url}{text}
    def link_handler(m):
        label = m.group(1)
        url = m.group(2)
        # Process label for bold/italic/etc but keep it simple
        processed_label = label
        processed_label = re.sub(r'\*\*([^\*]+)\*\*', r'\\textbf{\1}', processed_label)
        processed_label = re.sub(r'\*([^\*]+)\*', r'\\textit{\1}', processed_label)
        # Escape the label but not the URL
        return add_placeholder(r'\href{' + url + r'}{' + escape_latex(processed_label) + r'}')
    
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link_handler, text)
    
    # 4. Bold
    text = re.sub(r'\*\*([^\*]+)\*\*', lambda m: add_placeholder(r'\textbf{' + escape_latex(m.group(1)) + '}'), text)
    
    # 5. Italic
    text = re.sub(r'\*([^\*]+)\*', lambda m: add_placeholder(r'\textit{' + escape_latex(m.group(1)) + '}'), text)
    
    # 6. Escape the rest
    parts = re.split(r'(__PLACEHOLDER_\d+__)', text)
    res = ""
    for part in parts:
        m = re.match(r'__PLACEHOLDER_(\d+)__', part)
        if m:
            res += placeholders[int(m.group(1))]
        else:
            res += escape_latex(part)
    return res

def convert_set(file_list, output_filename, doc_title):
    preamble = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{xeCJK}
\usepackage[space]{grffile}
\usepackage{geometry}
\usepackage[inkscapelatex=false]{svg}
\usepackage{array}
\geometry{a4paper, margin=1in}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    xleftmargin=2em
}

\title{""" + doc_title + r"""}
\author{}
\date{2025年12月}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""
    postamble = r"\end{document}"
    
    content_list = [preamble]
    
    for filename in file_list:
        file_path = os.path.join(files_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found.")
            continue
            
        print(f"Processing {filename}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        state = {'list_stack': [], 'in_table': False, 'in_code_block': False, 'in_math_block': False}
        
        for i, line in enumerate(lines):
            line = line.rstrip('\n')
            
            # 1. Code blocks
            if line.strip().startswith('```'):
                if not state['in_code_block']:
                    state['in_code_block'] = True
                    content_list.append(r"\begin{lstlisting}" + "\n")
                else:
                    state['in_code_block'] = False
                    content_list.append(r"\end{lstlisting}" + "\n")
                continue
            
            if state['in_code_block']:
                content_list.append(line + "\n")
                continue

            # 2. Math blocks ($$ ... $$)
            # Check for starting a math block
            if not state['in_math_block'] and line.strip().startswith('$$'):
                if line.strip().endswith('$$') and len(line.strip()) >= 4:
                    # Single line display math
                    content_list.append(line + "\n")
                    continue
                else:
                    state['in_math_block'] = True
                    content_list.append(line + "\n")
                    continue
            
            # If we are inside a math block
            if state['in_math_block']:
                content_list.append(line + "\n")
                if line.strip().endswith('$$'):
                    state['in_math_block'] = False
                continue

            # Headings
            h_match = re.match(r'^(#+)\s+(.*)', line)
            if h_match:
                level_str, h_text = h_match.groups()
                # Clean up NVIDIA doc heading links
                h_text = re.sub(r'\s*\[#\].*$', '', h_text)
                # Strip redundant hardcoded numbers like "8.12. " or "1.1. "
                h_text = re.sub(r'^[\d\.]+\s+', '', h_text)

                level = len(level_str)
                if level == 1: cmd = "section"
                elif level == 2: cmd = "subsection"
                elif level == 3: cmd = "subsubsection"
                else: cmd = "paragraph"
                content_list.append(f"\\{cmd}{{{process_inline(h_text)}}}\n")
                continue


            # Images
            img_match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
            if img_match:
                alt, path = img_match.groups()
                path = path.strip('<>')
                if path.startswith('./'): path = path[2:]
                
                # Absolute path for images
                full_img_path = os.path.join(copied_images_dir, path).replace('\\', '/')
                
                content_list.append(r"\begin{figure}[h]\centering" + "\n")
                if path.endswith('.svg'):
                    content_list.append(f"\\includesvg[width=0.8\\textwidth]{{{full_img_path}}}\n")
                else:
                    content_list.append(f"\\includegraphics[width=0.8\\textwidth]{{{full_img_path}}}\n")
                content_list.append(f"\\caption{{{process_inline(alt)}}}\n\\end{{figure}}\n")
                continue

            # Tables
            table_match = False
            if '|' in line:
                # Heuristic: a table row has at least 2 pipes, and next/prev line might be separator
                if state['in_table']:
                    if re.match(r'^[|\s\-:]+$', line): # Separator
                        continue
                    table_match = True
                else:
                    # Check if next line is a separator
                    if i + 1 < len(lines) and re.match(r'^\s*\|?\s*[:\-]+\s*\|', lines[i+1]):
                        table_match = True

            # Close list if necessary
            list_match = re.match(r'^(\s*)([\*\-]|\d+\.)\s+(.*)', line)
            if state['list_stack'] and not list_match and line.strip() != "" and not table_match:
                while state['list_stack']:
                    _, ltype = state['list_stack'].pop()
                    content_list.append(f"\\end{{{ltype}}}\n")

            # Close table if necessary
            if state['in_table'] and not table_match:
                content_list.append("\\end{longtable}\n")
                state['in_table'] = False

            if list_match:
                indent_str, marker, content = list_match.groups()
                indent_size = len(indent_str)
                ltype = 'itemize' if marker in ['*', '-'] else 'enumerate'
                
                if not state['list_stack'] or indent_size > state['list_stack'][-1][0]:
                    state['list_stack'].append((indent_size, ltype))
                    content_list.append(f"\\begin{{{ltype}}}\n")
                elif indent_size < state['list_stack'][-1][0]:
                    while state['list_stack'] and indent_size < state['list_stack'][-1][0]:
                        _, old_ltype = state['list_stack'].pop()
                        content_list.append(f"\\end{{{old_ltype}}}\n")
                    if not state['list_stack'] or indent_size > state['list_stack'][-1][0]:
                        state['list_stack'].append((indent_size, ltype))
                        content_list.append(f"\\begin{{{ltype}}}\n")
                
                content_list.append(f"\\item {process_inline(content)}\n")
                continue

            if table_match:
                raw_cols = [c.strip() for c in line.strip('|').split('|')]
                cols = [process_inline(c) for c in raw_cols]
                if not state['in_table']:
                    state['in_table'] = True
                    # Estimate column count
                    content_list.append(f"\\begin{{longtable}}{{{'|'.join(['l']*len(cols))}}}\n\\hline\n")
                content_list.append(" & ".join(cols) + " \\\\ \\hline\n")
                continue

            if line.strip():
                content_list.append(process_inline(line) + "\n\n")
            else:
                content_list.append("\n")

        # End of file cleanup
        while state['list_stack']:
            _, ltype = state['list_stack'].pop()
            content_list.append(f"\\end{{{ltype}}}\n")
        if state['in_table']:
            content_list.append("\\end{longtable}\n")
        content_list.append("\n\\newpage\n")

    content_list.append(postamble)
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("".join(content_list))

if __name__ == "__main__":
    convert_set(original_files, "Tile_IR.tex", "Tile IR Specification")
    convert_set(translated_files, "Tile_IR_Translated.tex", "Tile IR 规范（中文翻译）")
