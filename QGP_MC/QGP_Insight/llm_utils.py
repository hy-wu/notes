import os
from openai import OpenAI
from dotenv import load_dotenv
import fitz
import json
import urllib.request
import xml.etree.ElementTree as ET
import re

# Load .env from the same directory as this script
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

def get_deepseek_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)

def extract_abstract(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i in range(min(3, len(doc))):
            text += doc[i].get_text()
        return text[:6000]
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def generate_summary(pdf_path):
    client = get_deepseek_client()
    if not client:
        return {"full": "API Key not found.", "short": "No Key"}
    
    content = extract_abstract(pdf_path)
    prompt = f"""
    你是一个顶尖的理论物理学家。请阅读以下物理论文内容，提供两个版本的中文综述：
    1. 【完整综述】：300-500字，包含研究背景、核心方法、主要结论。
    2. 【极简摘要】：20字以内，一句话说明核心物理贡献。
    
    输出格式必须是严格的 JSON：
    {{"full": "...", "short": "..."}}
    
    论文内容：
    {content}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a professional physics researcher. Always output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={ "type": "json_object" },
            stream=False
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"full": f"Error: {str(e)}", "short": "Error"}

def extract_full_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

from rank_bm25 import BM25Okapi
import jieba  # For Chinese tokenization if needed, but for physics English is common

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_rag_context(query, papers_db, top_k=5):
    all_chunks = []
    chunk_metadata = []
    
    for paper in papers_db:
        text = extract_full_text(paper["path"])
        if "Error" in text: continue
        
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(c)
            chunk_metadata.append({"filename": paper["filename"], "path": paper["path"]})
    
    if not all_chunks:
        return "", []

    # Simple tokenization by splitting and lowercasing (basic but effective for physics terms)
    tokenized_corpus = [doc.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split()
    top_n = bm25.get_top_n(tokenized_query, all_chunks, n=top_k)
    
    # Identify which papers were used
    used_papers = []
    context_text = ""
    for i, chunk in enumerate(top_n):
        # Find metadata for this chunk (this is a bit slow, but works for now)
        # In a real vector DB this is handled by IDs.
        idx = all_chunks.index(chunk)
        meta = chunk_metadata[idx]
        context_text += f"\n--- 来源: {meta['filename']} ---\n{chunk}\n"
        if meta['filename'] not in used_papers:
            used_papers.append(meta['filename'])
            
    return context_text, used_papers

def rag_answer(query, context):
    client = get_deepseek_client()
    if not client: return "未找到 API Key。"
    
    prompt = f"""
    你是一个理论物理学专家。请基于以下提供的【参考资料】回答用户的问题："{query}"。
    
    要求：
    1. 只能根据参考资料内容回答。如果资料中没有相关信息，请直接说明。
    2. 回答要专业、严谨，引用资料时请说明来源文件名。
    3. 优先使用中文回答，但物理术语可以保留英文。
    
    【参考资料】：
    {context}
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a senior physics researcher specialized in QGP and Monte Carlo simulations."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"RAG 回答出错: {str(e)}"

def semantic_search(query, summaries):
    client = get_deepseek_client()
    if not client: return "未找到 API Key。"
    
    prompt = f"""
    你是一个科研助手。用户正在检索 QGP 相关的文献，问题是："{query}"
    请根据以下已有的论文摘要数据库，分析并推荐最相关的论文。
    请列出文件名，并给出推荐理由。
    
    摘要数据库：
    {summaries}
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"搜索出错: {str(e)}"

import time

def search_arxiv(query, max_results=10, retries=2, delay=1):
    """
    Search for arXiv papers using Semantic Scholar as a more stable primary source, 
    falling back to direct arXiv API if needed.
    """
    # 1. Try Semantic Scholar (Stable, JSON, indexes arXiv)
    ss_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit={max_results}&fields=title,authors,year,abstract,externalIds,openAccessPdf"
    
    try:
        req = urllib.request.Request(ss_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            results = []
            for paper in data.get('data', []):
                # Filter for papers that have an arXiv ID
                ext_ids = paper.get('externalIds', {})
                arxiv_id = ext_ids.get('ArXiv')
                
                title = paper.get('title', 'No Title')
                summary = paper.get('abstract', 'No Summary')
                if summary is None: summary = "No Summary"
                
                authors = [a.get('name') for a in paper.get('authors', [])]
                year = paper.get('year', 'Unknown')
                
                # Determine PDF URL
                pdf_url = ""
                oa_pdf = paper.get('openAccessPdf')
                if oa_pdf:
                    pdf_url = oa_pdf.get('url')
                
                if not pdf_url and arxiv_id:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
                if arxiv_id or "qgp" in title.lower() or "qgp" in summary.lower():
                    results.append({
                        "title": title,
                        "summary": summary,
                        "authors": authors,
                        "published": str(year),
                        "pdf_url": pdf_url
                    })
            
            if results:
                return results
    except Exception as e:
        print(f"Semantic Scholar Search Error: {e}")

    # 2. Fallback to direct arXiv if Semantic Scholar fails or returns nothing
    # (Using direct connection with a conservative User-Agent)
    encoded_query = urllib.parse.quote(query)
    base_url = f'https://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending'
    
    for attempt in range(retries):
        try:
            time.sleep(delay * 2)
            req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            with urllib.request.urlopen(req, timeout=15) as response:
                xml_data = response.read().decode('utf-8')
                root = ET.fromstring(xml_data)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                results = []
                for entry in root.findall('atom:entry', ns):
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "No Title"
                    summary_elem = entry.find('atom:summary', ns)
                    summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else "No Summary"
                    authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
                    pub_elem = entry.find('atom:published', ns)
                    published = pub_elem.text[:10] if pub_elem is not None else "Unknown"
                    
                    pdf_url = ""
                    for link in entry.findall('atom:link', ns):
                        if link.get('title') == 'pdf' or link.get('type') == 'application/pdf':
                            pdf_url = link.get('href')
                    
                    if not pdf_url:
                        id_elem = entry.find('atom:id', ns)
                        if id_elem is not None:
                            pdf_url = id_elem.text.replace('abs', 'pdf') + ".pdf"

                    results.append({
                        "title": title,
                        "summary": summary,
                        "authors": authors,
                        "published": published,
                        "pdf_url": pdf_url
                    })
                return results
        except Exception as e:
            print(f"Direct arXiv Fallback Error (Attempt {attempt+1}): {e}")
            time.sleep(delay * 3)
            
    return []

def download_arxiv_pdf(pdf_url, save_path):
    try:
        # arXiv sometimes requires a User-Agent or it might fail/block
        req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        print(f"Download Error: {e}")
        return False
