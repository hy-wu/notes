import streamlit as st
import json
import os
import pandas as pd
from llm_utils import generate_summary, semantic_search
from scanner import scan_folders

st.set_page_config(page_title="QGP Paper Insight", layout="wide")

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stCard { 
        border-radius: 10px; 
        border: 1px solid #e1e4e8; 
        padding: 20px; 
        background: white;
        margin-bottom: 10px;
    }
    .stTitle { color: #1a73e8; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 QGP Literature Insight")
st.sidebar.header("Management")

def load_db():
    if not os.path.exists('papers_db.json'):
        return []
    with open('papers_db.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def save_db(db):
    with open('papers_db.json', 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=4, ensure_ascii=False)

db = load_db()

if st.sidebar.button("Re-scan Folders"):
    scan_folders()
    st.experimental_rerun()

# --- Search Section ---
st.subheader("Topic Search")
search_query = st.text_input("Ask a question about your papers:", placeholder="e.g. Which papers talk about local spin polarization?")
if search_query:
    summaries_text = "\n".join([f"File: {p['filename']}\nSummary: {p['summary']}" for p in db if p['summary']])
    with st.spinner("Analyzing..."):
        ans = semantic_search(search_query, summaries_text)
    st.info(ans)

# --- Library Section ---
st.subheader(f"Library ({len(db)} papers)")

# Filter
tag_list = sorted(list(set([t for p in db for t in p['tags']])))
selected_tag = st.multiselect("Filter by tags:", tag_list)

filtered_db = db
if selected_tag:
    filtered_db = [p for p in db if any(t in selected_tag for t in p['tags'])]

for i, paper in enumerate(filtered_db):
    with st.container():
        st.markdown(f'<div class="stCard">', unsafe_allow_html=True)
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"### {paper['filename']}")
            st.caption(f"Path: {paper['path']}")
            
            if paper['summary']:
                st.markdown("**AI Summary:**")
                st.write(paper['summary'])
            else:
                st.warning("No summary yet.")
        
        with cols[1]:
            if st.button("Generate Summary", key=f"sum_{i}"):
                with st.spinner("Processing..."):
                    s = generate_summary(paper['path'])
                    paper['summary'] = s
                    save_db(filtered_db) # Update main db too
                    st.experimental_rerun()
            
            # Local preview link
            link = f"file:///{paper['path']}"
            st.markdown(f"[Open File]({link})")
        st.markdown('</div>', unsafe_allow_html=True)
