import os
from openai import OpenAI
from dotenv import load_dotenv
import fitz
import json

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
