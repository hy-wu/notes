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
        data = json.load(f)
        # 兼容性修复：确保旧数据也有 short_summary 字段
        for p in data:
            if "short_summary" not in p: p["short_summary"] = None
        return data

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4, ensure_ascii=False)

db = load_db()

papers_db = [p for p in db if not ("box_" in p["filename"] and "cuda_out" in p["path"])]
data_db = [p for p in db if "box_" in p["filename"] and "cuda_out" in p["path"]]

with st.sidebar:
    st.title("⚙️ 控制面板")
    if st.button("🔄 重新扫描目录", use_container_width=True):
        scan_folders()
        st.rerun()
    
    st.divider()
    st.subheader("🚀 自动化任务")
    pending = [p for p in papers_db if not p["summary"]]
    batch_size = 5
    if st.button(f"批量生成 {batch_size} 篇摘要 (剩余 {len(pending)} 篇)", use_container_width=True, disabled=len(pending)==0):
        to_process = pending[:batch_size]
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, paper in enumerate(to_process):
            status_text.text(f"正在分析 ({i+1}/{len(to_process)}): {paper['filename']}")
            res = generate_summary(paper["path"])
            
            # 更新内存和数据库
            paper["summary"] = res.get("full")
            paper["short_summary"] = res.get("short")
            for p in db:
                if p["path"] == paper["path"]:
                    p["summary"] = paper["summary"]
                    p["short_summary"] = paper["short_summary"]
                    break
            
            progress_bar.progress((i + 1) / len(to_process))
        
        save_db(db)
        st.success(f"已完成 {len(to_process)} 篇论文的批量分析！")
        st.rerun()

    st.divider()
    st.subheader("📊 库统计")
    st.write(f"学术文献: {len(papers_db)} 篇")
    st.write(f"模拟结果: {len(data_db)} 份")
    sum_count = len([p for p in papers_db if p["summary"]])
    st.progress(sum_count/len(papers_db) if papers_db else 0)
    st.write(f"摘要生成率: {sum_count}/{len(papers_db)}")

# --- Header ---
st.title("🔬 QGP Literature Insight")

# --- Caching the Search to save Tokens and Time ---
@st.cache_data(show_spinner=False)
def cached_semantic_search(query, text):
    return semantic_search(query, text)

# --- Search Section ---
st.subheader("🔍 智能检索")
search_query = st.text_input("输入话题、公式或关键词：", placeholder="例如：哪些论文提到了 Wigner 函数？")

if search_query:
    valid_summaries = [p for p in papers_db if p["summary"]]
    summaries_text = "\n".join([f"文件: {p['filename']}\n摘要: {p['summary']}" for p in valid_summaries])
    
    if summaries_text:
        # 使用 with 确保状态在结束后能正确更新
        with st.status(f"🚀 正在分析 {len(valid_summaries)} 篇论文摘要...", expanded=True) as status:
            ans = cached_semantic_search(search_query, summaries_text)
            status.update(label="✅ AI 语义分析完成", state="complete", expanded=False)
        st.chat_message("assistant").write(ans)
    else:
        st.warning("💡 **当前处于关键词检索模式**。由于你还没有生成任何论文摘要，AI 无法进行深度语义分析。建议你先点击下方论文的“生成摘要”按钮，或者在侧边栏点击“批量生成”。")
        
        # 本地匹配结果预览
        matches = [p for p in papers_db if search_query.lower() in p["filename"].lower()]
        if matches:
            st.success(f"找到 {len(matches)} 个标题匹配的文件。请在下方的“文献库”标签页中查看。")
        else:
            st.error("未找到匹配的文件名。")
tab1, tab2, tab3 = st.tabs(["📚 文献库", "🧩 知识集群", "💾 模拟数据结果"])

with tab1:
    all_tags = sorted(list(set([t for p in papers_db for t in p["tags"]])))
    selected_tags = st.multiselect("按关键词标签过滤:", all_tags)
    
    filtered_db = papers_db
    if selected_tags:
        filtered_db = [p for p in papers_db if any(t in selected_tags for t in p["tags"])]
    elif search_query:
        filtered_db = [p for p in papers_db if search_query.lower() in p["filename"].lower()]

    for i, paper in enumerate(filtered_db):
        # 构造标题栏：文件名 + 标签 + 20字概要
        tag_str = f"[{', '.join(paper['tags'])}]" if paper['tags'] else ""
        short_sum = f" | {paper['short_summary']}" if paper['short_summary'] else ""
        expander_title = f"📄 {paper['filename']}  {tag_str}{short_sum}"
        
        with st.expander(expander_title, expanded=False):
            col1, col2 = st.columns([4, 1])
            with col1:
                if paper["summary"]:
                    st.markdown("**AI 详细概述 (DeepSeek):**")
                    st.info(paper["summary"])
                else:
                    st.write("*待生成摘要...*")
                st.caption(f"路径: {paper['path']}")
            
            with col2:
                if st.button("生成摘要", key=f"btn_{i}"):
                    with st.spinner("DeepSeek 正在阅读并总结..."):
                        res = generate_summary(paper["path"])
                        # 更新当前数据
                        paper["summary"] = res.get("full")
                        paper["short_summary"] = res.get("short")
                        # 同步到数据库
                        for p in db:
                            if p["path"] == paper["path"]:
                                p["summary"] = paper["summary"]
                                p["short_summary"] = paper["short_summary"]
                                break
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
    sorted_data = sorted(data_db, key=lambda x: x["filename"])
    for d in sorted_data:
        cols = st.columns([5, 1])
        cols[0].write(f"📊 `{d['filename']}`")
        cols[1].markdown(f"[查看结果](file:///{d['path']})")
