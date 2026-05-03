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
            
        # state['list_stack'] stores (indent_size, list_type)
        state = {'list_stack': [], 'in_table': False}
        
        for line in lines:
            line = line.rstrip('\n')
            
            # Skip first title if it matches the title we just added (loosely)
            if line.strip().lower() == title.lower() or line.strip().lower() == filename.split('.')[0].lower():
                continue

            # Check for matches
            h_match = re.match(r'^(#+)\s+(.*)', line)
            img_match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
            list_match = re.match(r'^(\s*)([\*\-]|\d+\.)\s+(.*)', line)
            table_match = '|' in line and not line.strip().startswith('---')

            # List Handling with Nesting
            if list_match:
                indent_str, marker, content = list_match.groups()
                indent_size = len(indent_str)
                ltype = 'itemize' if marker in ['*', '-'] else 'enumerate'
                
                # If we are deeper than current stack
                if not state['list_stack'] or indent_size > state['list_stack'][-1][0]:
                    state['list_stack'].append((indent_size, ltype))
                    output.append(f"\\begin{{{ltype}}}\n")
                # If we are shallower
                elif indent_size < state['list_stack'][-1][0]:
                    while state['list_stack'] and indent_size < state['list_stack'][-1][0]:
                        old_indent, old_ltype = state['list_stack'].pop()
                        output.append(f"\\end{{{old_ltype}}}\n")
                    
                    if not state['list_stack'] or indent_size > state['list_stack'][-1][0]:
                        state['list_stack'].append((indent_size, ltype))
                        output.append(f"\\begin{{{ltype}}}\n")
                    elif state['list_stack'][-1][1] != ltype:
                        # Same indent but different type
                        old_indent, old_ltype = state['list_stack'].pop()
                        output.append(f"\\end{{{old_ltype}}}\n")
                        state['list_stack'].append((indent_size, ltype))
                        output.append(f"\\begin{{{ltype}}}\n")
                # Same level
                elif state['list_stack'][-1][1] != ltype:
                    old_indent, old_ltype = state['list_stack'].pop()
                    output.append(f"\\end{{{old_ltype}}}\n")
                    state['list_stack'].append((indent_size, ltype))
                    output.append(f"\\begin{{{ltype}}}\n")
                
                output.append(f"\\item {process_inline(content)}\n")
                continue
            
            # If not a list match, but we are in a list
            if state['list_stack'] and line.strip() != "":
                # Check if it's an image or something that should stay inside or close
                if img_match or h_match:
                    # Close all lists
                    while state['list_stack']:
                        old_indent, old_ltype = state['list_stack'].pop()
                        output.append(f"\\end{{{old_ltype}}}\n")
                elif not table_match:
                    # It might be a continuation of a list item if it's indented
                    first_char_indent = len(line) - len(line.lstrip())
                    if first_char_indent > state['list_stack'][-1][0]:
                        output.append(process_inline(line.strip()) + "\n")
                        continue
                    else:
                        # Close lists
                        while state['list_stack']:
                            old_indent, old_ltype = state['list_stack'].pop()
                            output.append(f"\\end{{{old_ltype}}}\n")

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
        while state['list_stack']:
            old_indent, old_ltype = state['list_stack'].pop()
            output.append(f"\\end{{{old_ltype}}}\n")
        if state['in_table']:
            output.append("\\end{longtable}\n")
            
        output.append("\n\\newpage\n")



    output.append(LATEX_POSTAMBLE)
    
    with open('CUDA_Notes.tex', 'w', encoding='utf-8') as f:
        f.write("".join(output))
    print("Generated CUDA_Notes.tex")

if __name__ == "__main__":
    convert_md_to_tex()
