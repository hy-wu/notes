import os
import re

files_and_titles = [
    ("p4arch.md", "GPU Architecture"),
    ("p5mem.md", "GPU Memory"),
    ("p6perf.md", "Performance Metrics"),
    ("p8conv.md", "Convolution"),
    ("p9sten.md", "Stencil Computation"),
    ("p10reduc.md", "Reduction"),
    ("p11scanKS.md", "Scan (Kogge-Stone)"),
    ("p12scanBK.md", "Scan (Brent-Kung)"),
    ("p13hist.md", "Histogram"),
    ("p14merge.md", "Merge"),
    ("p15sort.md", "Sort"),
    ("p16COO_CSR.md", "Sparse Matrix (COO, CSR)"),
    ("p17ELL_JDS.md", "Sparse Matrix (ELL, JDS)"),
    ("p18graph1.md", "Graph Algorithms I"),
    ("p19graph2.md", "Graph Algorithms II"),
    ("p20warp_sync.md", "Warp Synchronization"),
    ("p21pinnd_mem_stream.md", "Pinned Memory and Streams"),
    ("p22dynParal.md", "Dynamic Parallelism"),
    ("p23potpourri.md", "CUDA Potpourri"),
    ("p24matmul.md", "Matrix Multiplication Optimization"),
]

LATEX_PREAMBLE = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
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
\geometry{a4paper, margin=1in}

\title{CUDA Programming Notes (CMPS 297 GPU)}
\author{Combined Notes}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""

LATEX_POSTAMBLE = r"""
\end{document}
"""

def escape_latex(text, in_math=False):
    if in_math:
        return text
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

    # 1. Math
    text = re.sub(r'\$.*?\$', lambda m: add_placeholder(m.group(0)), text)
    
    # 2. Code
    text = re.sub(r'`([^`]+)`', lambda m: add_placeholder(r'\texttt{' + escape_latex(m.group(1)) + '}'), text)
    
    # 3. Bold
    text = re.sub(r'\*\*([^\*]+)\*\*', lambda m: add_placeholder(r'\textbf{' + escape_latex(m.group(1)) + '}'), text)
    
    # 4. Italic
    text = re.sub(r'\*([^\*]+)\*', lambda m: add_placeholder(r'\textit{' + escape_latex(m.group(1)) + '}'), text)
    
    # 5. Escape the rest
    parts = re.split(r'(__PLACEHOLDER_\d+__)', text)
    res = ""
    for part in parts:
        m = re.match(r'__PLACEHOLDER_(\d+)__', part)
        if m:
            res += placeholders[int(m.group(1))]
        else:
            res += escape_latex(part)
    return res

def convert_md_to_tex():
    output = [LATEX_PREAMBLE]
    
    for filename, title in files_and_titles:
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found.")
            continue
            
        output.append(f"\\section{{{title}}}\n")
        
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        state = {'in_list': False, 'in_table': False, 'list_type': None}
        
        for line in lines:
            line = line.rstrip('\n')
            
            # Skip first title if it matches the title we just added (loosely)
            if line.strip().lower() == title.lower() or line.strip().lower() == filename.split('.')[0].lower():
                continue

            # Check for matches
            h_match = re.match(r'^(#+)\s+(.*)', line)
            img_match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
            list_match = re.match(r'^(\s*)([\*\-]|\d+\.)\s+(.*)', line)
            table_match = '|' in line and not line.strip().startswith('---') # Basic table check

            # Close list if necessary
            if state['in_list'] and not list_match and line.strip() != "":
                output.append(f"\\end{{{state['list_type']}}}\n")
                state['in_list'] = False
            
            # Close table if necessary
            if state['in_table'] and not table_match:
                output.append("\\end{longtable}\n")
                state['in_table'] = False

            if h_match:
                level, h_text = h_match.groups()
                tex_cmd = "subsection" if len(level) == 1 else "subsubsection"
                output.append(f"\\{tex_cmd}{{{process_inline(h_text)}}}\n")
                continue

            if img_match:
                alt, path = img_match.groups()
                path = path.strip('<>')
                output.append(f"\\begin{{figure}}[h]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{{{path}}}\n\\caption{{{process_inline(alt)}}}\n\\end{{figure}}\n")
                continue

            if list_match:
                indent, marker, content = list_match.groups()
                ltype = 'itemize' if marker in ['*', '-'] else 'enumerate'
                if not state['in_list']:
                    state['in_list'] = True
                    state['list_type'] = ltype
                    output.append(f"\\begin{{{ltype}}}\n")
                elif state['list_type'] != ltype:
                    output.append(f"\\end{{{state['list_type']}}}\n")
                    state['list_type'] = ltype
                    output.append(f"\\begin{{{ltype}}}\n")
                output.append(f"\\item {process_inline(content)}\n")
                continue

            if table_match:
                if re.match(r'^[|\s\-:]+$', line):
                    continue
                raw_cols = line.strip('|').split('|')
                cols = [process_inline(c.strip()) for c in raw_cols]
                
                if not state['in_table']:
                    state['in_table'] = True
                    output.append(f"\\begin{{longtable}}{{{'|'.join(['l']*len(cols))}}}\n\\hline\n")
                
                output.append(" & ".join(cols) + " \\\\ \\hline\n")
                continue

            # Normal text
            if line.strip():
                output.append(process_inline(line) + "\n\n")
            else:
                output.append("\n")

        # Clean up states at end of file
        if state['in_list']:
            output.append(f"\\end{{{state['list_type']}}}\n")
        if state['in_table']:
            output.append("\\end{longtable}\n")
            
        output.append("\n\\newpage\n")


    output.append(LATEX_POSTAMBLE)
    
    with open('CUDA_Notes.tex', 'w', encoding='utf-8') as f:
        f.write("".join(output))
    print("Generated CUDA_Notes.tex")

if __name__ == "__main__":
    convert_md_to_tex()
