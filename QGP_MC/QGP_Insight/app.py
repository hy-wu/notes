import streamlit as st
import json
import os
from llm_utils import generate_summary, semantic_search
from scanner import scan_folders

st.set_page_config(page_title="QGP Paper Insight", layout="wide", page_icon="🔬")

DB_PATH = r"C:\Users\hy-wu.DESKTOP-G355NC5\Documents\GitHub\notes\QGP_MC\QGP_Insight\papers_db.json"

def load_db():
    if not os.path.exists(DB_PATH): return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4, ensure_ascii=False)

db = load_db()

# --- Pre-processing: Separate Literature and Simulation Data ---
# 识别规则：文件名包含 box_ 且路径包含 cuda_out 的视为模拟数据
papers_db = [p for p in db if not ("box_" in p["filename"] and "cuda_out" in p["path"])]
data_db = [p for p in db if "box_" in p["filename"] and "cuda_out" in p["path"]]

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ 控制面板")
    if st.button("🔄 重新扫描目录", use_container_width=True):
        scan_folders()
        st.rerun()
    
    st.divider()
    st.subheader("📊 统计信息")
    st.write(f"学术文献: {len(papers_db)} 篇")
    st.write(f"模拟结果: {len(data_db)} 份")
    sum_count = len([p for p in papers_db if p['summary']])
    st.progress(sum_count/len(papers_db) if papers_db else 0)
    st.write(f"摘要生成率: {sum_count}/{len(papers_db)}")

# --- Header ---
st.title("🔬 QGP Literature Insight")

# --- Search Section ---
search_query = st.text_input("🔍 智能检索 (输入话题、公式或关键词)", placeholder="例如：自旋极化中的 Wigner 函数演化...")

if search_query:
    summaries_text = "\n".join([f"File: {p['filename']}\nSummary: {p['summary']}" for p in papers_db if p['summary']])
    if summaries_text:
        with st.spinner("DeepSeek 正在思考..."):
            ans = semantic_search(search_query, summaries_text)
        st.chat_message("assistant").write(ans)

# --- Library Tabs ---
tab1, tab2, tab3 = st.tabs(["📚 文献库", "🧩 知识集群", "💾 模拟数据结果"])

with tab1:
    # Filter
    all_tags = sorted(list(set([t for p in papers_db for t in p["tags"]])))
    selected_tags = st.multiselect("按关键词标签过滤:", all_tags)
    
    filtered_db = papers_db
    if selected_tags:
        filtered_db = [p for p in papers_db if any(t in selected_tags for t in p["tags"])]
    elif search_query:
        filtered_db = [p for p in papers_db if search_query.lower() in p["filename"].lower()]

    for i, paper in enumerate(filtered_db):
        with st.expander(f"📄 {paper['filename']}", expanded=(i==0)):
            col1, col2 = st.columns([4, 1])
            with col1:
                # 修复 Tag 显示：使用 Streamlit 原生组件确保可见性
                if paper['tags']:
                    st.write("标签: " + " ".join([f"`{t}`" for t in paper['tags']]))
                
                if paper['summary']:
                    st.markdown("**AI 概述 (DeepSeek-V3):**")
                    st.info(paper['summary'])
                else:
                    st.write("*待生成摘要...*")
                st.caption(f"路径: {paper['path']}")
            
            with col2:
                if st.button("生成摘要", key=f"btn_{i}"):
                    with st.spinner("正在阅读 PDF..."):
                        paper['summary'] = generate_summary(paper['path'])
                        # 在原始 db 中更新
                        for p in db:
                            if p['path'] == paper['path']:
                                p['summary'] = paper['summary']
                        save_db(db)
                        st.rerun()
                st.markdown(f"[📂 打开本地文件](file:///{paper['path']})")

with tab2:
    groups = {
        "Spin & Vorticity Theory": [p for p in papers_db if "spin" in p["tags"] or "vorticity" in p["tags"]],
        "Monte Carlo & Kinetic": [p for p in papers_db if "monte" in p["tags"] or "mc" in p["tags"]],
        "Reference Books": [p for p in papers_db if "books" in p["path"]]
    }
    for gname, gfiles in groups.items():
        if gfiles:
            with st.expander(f"{gname} ({len(gfiles)} 篇)"):
                for f in gfiles:
                    st.write(f"- {f['filename']}")

with tab3:
    st.subheader("Cuda 模拟结果记录")
    st.write("这些是自动识别出的 `box_...` 类型数据结果文件。")
    # 按截面参数 d 排序展示
    sorted_data = sorted(data_db, key=lambda x: x['filename'])
    for d in sorted_data:
        cols = st.columns([5, 1])
        cols[0].write(f"📊 `{d['filename']}`")
        cols[1].markdown(f"[查看结果](file:///{d['path']})")
