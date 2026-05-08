import streamlit as st
import json
import os
from llm_utils import generate_summary, semantic_search, search_arxiv, download_arxiv_pdf, extract_full_text, get_rag_context, rag_answer
from scanner import scan_folders

st.set_page_config(page_title="QGP Paper Insight", layout="wide", page_icon="🔬")

DB_PATH = r"C:\Users\hy-wu.DESKTOP-G355NC5\Documents\GitHub\notes\QGP_MC\QGP_Insight\papers_db.json"
DOWNLOAD_DIR = r"C:\Users\hy-wu.DESKTOP-G355NC5\Downloads"

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
st.subheader("🔍 RAG 深度文献问答")
col_q1, col_q2 = st.columns([4, 1])
search_query = col_q1.text_input("输入具体的物理问题、公式或课题：", placeholder="例如：描述 BAMPS 框架中使用的碰撞积分项。")
rag_mode = col_q2.selectbox("检索策略", ["摘要语义 (快)", "全文混合 (准)", "关键词定位"])

if search_query:
    if rag_mode == "全文混合 (准)":
        with st.status("🚀 正在跨文献检索相关片段...", expanded=True) as status:
            context, used_papers = get_rag_context(search_query, papers_db)
            if context:
                status.update(label=f"✅ 找到来自 {len(used_papers)} 篇论文的相关内容，正在合成回答...", state="running")
                ans = rag_answer(search_query, context)
                status.update(label="✅ RAG 深度分析完成", state="complete", expanded=False)
                st.chat_message("assistant").write(ans)
                with st.expander("查看参考原文片段"):
                    st.text(context)
            else:
                status.update(label="❌ 未找到相关内容", state="complete", expanded=False)
                st.error("无法在库中找到与该问题相关的具体片段。")
                
    elif rag_mode == "关键词定位":
        with st.status("🚀 正在全文搜索关键词...", expanded=True) as status:
            results = []
            for p in papers_db:
                full_text = extract_full_text(p["path"])
                if search_query.lower() in full_text.lower():
                    idx = full_text.lower().find(search_query.lower())
                    start = max(0, idx - 200)
                    end = min(len(full_text), idx + 200)
                    context = full_text[start:end].replace('\n', ' ')
                    results.append({"filename": p["filename"], "context": context})
            
            if results:
                status.update(label=f"✅ 在 {len(results)} 篇论文中找到匹配", state="complete", expanded=False)
                for r in results:
                    with st.expander(f"📄 {r['filename']}"):
                        st.markdown(f"...{r['context']}...")
            else:
                status.update(label="❌ 未找到匹配", state="complete", expanded=False)

    else:  # 摘要语义
        valid_summaries = [p for p in papers_db if p["summary"]]
        summaries_text = "\n".join([f"文件: {p['filename']}\n摘要: {p['summary']}" for p in valid_summaries])
        
        if summaries_text:
            with st.status(f"🚀 正在分析 {len(valid_summaries)} 篇论文摘要...", expanded=True) as status:
                ans = cached_semantic_search(search_query, summaries_text)
                status.update(label="✅ AI 语义分析完成", state="complete", expanded=False)
            st.chat_message("assistant").write(ans)
        else:
            st.warning("💡 请先生成摘要以使用语义检索模式。")

tab1, tab2, tab3, tab4 = st.tabs(["📚 文献库", "🧩 知识集群", "💾 模拟数据结果", "🌐 学术查新"])

with tab1:
    col_f1, col_f2 = st.columns([2, 1])
    all_tags = sorted(list(set([t for p in papers_db for t in p["tags"]])))
    selected_tags = col_f1.multiselect("按关键词标签过滤:", all_tags)
    sort_by = col_f2.selectbox("排序方式:", ["最近扫描", "文件名 (A-Z)", "已生成摘要优先"])
    
    filtered_db = papers_db
    if selected_tags:
        filtered_db = [p for p in papers_db if any(t in selected_tags for t in p["tags"])]
    elif search_query and not (search_query and "rag_mode" in locals() and rag_mode != "摘要语义 (快)"):
        filtered_db = [p for p in papers_db if search_query.lower() in p["filename"].lower()]

    # Apply Sorting
    if sort_by == "最近扫描":
        filtered_db = sorted(filtered_db, key=lambda x: x.get("scanned_at", ""), reverse=True)
    elif sort_by == "文件名 (A-Z)":
        filtered_db = sorted(filtered_db, key=lambda x: x["filename"].lower())
    elif sort_by == "已生成摘要优先":
        filtered_db = sorted(filtered_db, key=lambda x: (x["summary"] is None, x["filename"].lower()))

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

with tab4:
    st.subheader("🌐 全网学术文献搜索 (arXiv & Semantic Scholar)")
    col_s1, col_s2 = st.columns([4, 1])
    arxiv_query = col_s1.text_input("搜索论文 (标题/作者/物理关键词):", placeholder="例如: Quark Gluon Plasma Spin")
    max_res = col_s2.slider("结果数量", 5, 50, 10)
    
    if arxiv_query:
        with st.spinner(f"正在全网检索 '{arxiv_query}'..."):
            results = search_arxiv(arxiv_query, max_results=max_res)
            
        if not results:
            st.warning("未找到相关文献。")
        else:
            st.success(f"找到 {len(results)} 篇相关文献 (已整合多源数据)：")
            for i, res in enumerate(results):
                source_tag = " [arXiv]" if "arxiv.org" in res['pdf_url'].lower() else " [Academic]"
                with st.expander(f"🆕 [{res['published']}] {res['title']}{source_tag}"):
                    st.markdown(f"**作者:** {', '.join(res['authors'])}")
                    st.markdown(f"**摘要:** {res['summary']}")
                    
                    # 清理文件名
                    safe_title = "".join([c if c.isalnum() or c in ' .-_' else '_' for c in res['title']])
                    # 限制长度
                    safe_title = safe_title[:100]
                    save_name = f"{safe_title}.pdf"
                    save_path = os.path.join(DOWNLOAD_DIR, save_name).replace('\\', '/')
                    
                    c1, c2 = st.columns([1, 1])
                    if c1.button("📥 下载并导入库", key=f"dl_{i}"):
                        with st.spinner(f"正在下载并导入: {save_name}..."):
                            if download_arxiv_pdf(res['pdf_url'], save_path):
                                st.success("下载成功！正在重新扫描目录...")
                                scan_folders()
                                st.rerun()
                            else:
                                st.error("下载失败，请检查网络连接。")
                    if res['pdf_url']:
                        c2.markdown(f"[🔗 查看源文件]({res['pdf_url']})")
                    else:
                        c2.write("*(未找到直接 PDF 链接)*")
