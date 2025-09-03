import streamlit as st
from datetime import datetime
import pandas as pd

from utils import extract_text_from_uploaded_file, chunk_text, get_embeddings_batch, generate_answer
from vector_store import SimpleVectorStore


# Page config and SEO metadata
st.set_page_config(
    page_title="StudyMate AI - Smart Study Helper",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <meta name="description" content="StudyMate AI lets you upload PDFs, DOCX, PPTX, and TXT files and quickly get answers.">
    <meta name="keywords" content="AI, StudyMate, PDFs, DOCX, PPTX, TXT, Question Answering">
""", unsafe_allow_html=True)


# Custom CSS for white background and polished UI
st.markdown("""
<style>
body { background-color: #ffffff; }
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.feature-card {
    background: #f8fafc;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}
.answer-box {
    background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    margin: 1rem 0;
}
.source-box {
    background: #e2e8f0;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state variables
if "vs" not in st.session_state:
    st.session_state.vs = SimpleVectorStore()
if "documents" not in st.session_state:
    st.session_state.documents = []


# Header
st.markdown("""
<div class="main-header">
    <h1>✍️ StudyMate AI</h1>
    <p>Upload PDFs, DOCX, PPTX, or TXT and ask questions instantly</p>
</div>
""", unsafe_allow_html=True)


# Sidebar for actions and feature info
st.sidebar.markdown("### 📋 Actions")
if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.vs.reset()
    st.session_state.documents = []
    st.sidebar.success("Cleared all in-memory data")

st.sidebar.markdown("""
### Core Features
- 📄 Multi-file support (PDF, DOCX, PPTX, TXT)  
- 🔍 Text extraction  
- 🤖 Embeddings for semantic search  
- 💬 Question answering  
- 🎯 Polished UI with white background  
""")


# Loader function to show while processing
def show_loader(message="Processing..."):
    loader_placeholder = st.empty()
    loader_placeholder.markdown(f"""
        <div style="
            display:flex; 
            justify-content:center; 
            align-items:center; 
            height:250px; 
            background-color:#e6f7ff;
            border-radius:12px;
            flex-direction:column;
        ">
            <img src="https://i.gifer.com/ZZ5H.gif" width="120" />
            <p style="font-size:18px; font-weight:bold; margin-top:12px;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
    return loader_placeholder


# File upload and processing
uploaded_files = st.file_uploader(
    "Upload files (PDF, DOCX, PPTX, TXT)",
    accept_multiple_files=True,
    type=["pdf", "docx", "pptx", "txt"]
)

if uploaded_files and st.button("🚀 Process Files"):
    total_files = len(uploaded_files)
    processed = 0
    loader = show_loader("Processing uploaded files...")

    for uf in uploaded_files:
        st.info(f"Processing {uf.name}")
        try:
            text, ftype = extract_text_from_uploaded_file(uf)
        except Exception as e:
            st.error(f"Error extracting text from {uf.name}: {e}")
            continue

        if not text:
            st.warning(f"No text found in {uf.name}")
            continue

        chunks = chunk_text(text)

        try:
            embeddings = get_embeddings_batch(chunks, doc_name=uf.name)
        except Exception as e:
            st.error(f"Error generating embeddings for {uf.name}: {e}")
            continue

        metadatas = [{"document": uf.name, "chunk_index": idx, "text": chunk} for idx, chunk in enumerate(chunks)]

        try:
            st.session_state.vs.add(embeddings, metadatas)
        except Exception as e:
            st.error(f"Error adding vectors for {uf.name}: {e}")
            continue

        st.session_state.documents.append({
            "name": uf.name,
            "length": len(text),
            "chunks": len(chunks),
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        processed += 1

    loader.empty()
    st.success(f"Processed {processed}/{total_files} files")


# Show processed documents and Q&A interface
if st.session_state.documents:
    st.subheader("📄 Processed Documents")
    df = pd.DataFrame(st.session_state.documents)
    st.dataframe(df, use_container_width=True)

    st.subheader("💬 Ask a Question")
    question = st.text_area("Enter your question here", height=120)
    top_k = st.slider("Number of retrieved chunks (top_k)", 1, 10, 4)

    if st.button("🔍 Get Answer"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            loader = show_loader("Searching relevant passages...")
            try:
                q_emb = get_embeddings_batch([question])[0]
                hits = st.session_state.vs.search(q_emb, top_k=top_k)
            except Exception as e:
                st.error(f"Search error: {e}")
                hits = []
            loader.empty()

            if not hits:
                st.warning("No relevant passages found. Please upload documents first.")
            else:
                contexts = [{"text": h["metadata"]["text"], "source": h["metadata"].get("document", "unknown")} for h in hits]

                loader = show_loader("Generating answer...")
                try:
                    answer = generate_answer(question, contexts)
                except Exception as e:
                    st.error(f"Answer generation error: {e}")
                    answer = "Error: Unable to generate answer."
                loader.empty()

                st.markdown(f"""
                <div class="answer-box">
                    <h3>🤖 Answer</h3>
                    <p>{answer}</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📖 Sources")
                for i, h in enumerate(hits, 1):
                    md = h["metadata"]
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong> {md.get('document')} — score {h['score']:.3f}<br>
                        {md['text'][:400]}{"…" if len(md['text']) > 400 else ""}
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 How StudyMate Works</h3>
        <ol>
            <li><strong>Upload Files:</strong> PDFs, DOCX, PPTX, TXT</li>
            <li><strong>Ask Questions:</strong> Type natural language queries</li>
            <li><strong>Get Answers:</strong> Based on extracted content</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.info("👆 Start by uploading your documents above!")
