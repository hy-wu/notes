import os
from openai import OpenAI
from dotenv import load_dotenv
import fitz

# Load .env from the same directory as this script
env_path = os.path.join(os.path.dirname(__file__), '.env')
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
        # Extract first 3 pages
        for i in range(min(3, len(doc))):
            text += doc[i].get_text()
        return text[:6000] # DeepSeek has large context, 6000 chars is safe
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def generate_summary(pdf_path):
    client = get_deepseek_client()
    if not client:
        return "API Key not found. Please set DEEPSEEK_API_KEY in .env file."
    
    content = extract_abstract(pdf_path)
    prompt = f"""
    你是一个顶尖的理论物理学家，擅长高能重离子物理（QGP）、QCD以及蒙特卡洛算法。
    请仔细阅读以下物理论文的摘要和引言部分，并提供一个专业、精准的中文综述（300-500字）。
    
    综述结构：
    1. 【研究背景】：说明该研究试图解决什么物理问题。
    2. 【核心方法】：详细描述所采用的理论模型或数值算法（如 Boltzmann 演化、自旋极化计算等）。
    3. 【主要结论】：概括其物理意义。
    
    论文内容：
    {content}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # deepseek-v3/v2.5
            messages=[
                {"role": "system", "content": "You are a professional research assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"DeepSeek API Error: {str(e)}"

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
