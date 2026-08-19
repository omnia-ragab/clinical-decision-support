import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import pandas as pd

# ==========================================
# 1. Page Configuration & Custom CSS (UI/UX)
# ==========================================
st.set_page_config(page_title="Clinical Decision Support", layout="wide", initial_sidebar_state="expanded")

# CSS مخصص لعمل الـ Sidebar باللون الأزرق الغامق وتنسيق الكروت
st.markdown("""
<style>
    /* Dark Blue Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F2537 !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Buttons in sidebar */
    .stButton>button {
        border-radius: 8px;
        text-align: left;
    }
    /* Evidence Box Styling */
    .evidence-box {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1A73E8;
        margin-bottom: 15px;
        font-size: 0.9em;
    }
    /* Score Badge */
    .score-badge {
        background-color: #CEEAD6;
        color: #0D652D;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Sidebar Navigation
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ Clinical Decision Support ᴸᴵᵀᴱ")
    st.markdown("---")
    st.button("💬 Ask Question", use_container_width=True)
    st.button("🕒 History", use_container_width=True)
    st.button("📚 Sources", use_container_width=True)
    st.button("ℹ️ About", use_container_width=True)
    st.markdown("---")
    st.button("⚙️ Settings", use_container_width=True)
    st.button("🚪 Logout", use_container_width=True)

# ==========================================
# 3. Model & DB Initialization
# ==========================================
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
except KeyError:
    st.error("Please set GEMINI_API_KEY in Streamlit Secrets!")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(name="who_hypertension_guideline_v2_cosine")

embedding_model = load_embedding_model()
collection = load_chroma_collection()

def normalize_embedding(emb):
    vec = np.array(emb)
    norm = np.linalg.norm(vec)
    if norm > 0:
        return (vec / norm).tolist()
    return vec.tolist()

def safe_query(question, top_k=3):
    raw_emb = embedding_model.encode([question])[0]
    query_embedding = normalize_embedding(raw_emb)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    best_distance = results["distances"][0][0] if results["distances"][0] else 1.0
    
    if best_distance > 0.35: 
        return {"in_scope": False, "retrieved_chunks": results}
        
    return {"in_scope": True, "retrieved_chunks": results}

RAG_SYSTEM_PROMPT = """
You are a clinical decision support AI acting as an evidence synthesizer.
Your ONLY source of truth is the provided clinical guidelines context.

CORE PHILOSOPHY: Fluent -> Safe.

RULES:
1. If the context contains the answer, summarize it concisely under "Recommendation", followed by "Supporting Evidence" with bullet points.
2. CITATIONS ARE MANDATORY: Every claim must end with a citation strictly in this format: [Document Name - Section: <Section Name> - Page <Page Number> - Chunk <Chunk ID>].
3. If the user asks for personal medical advice, opinions outside the context, or out-of-scope questions, you MUST refuse using EXACTLY this 3-part structure:
   "1. Insufficiency: The provided WHO hypertension guideline does not contain data or recommendations to address your specific question.
    2. Honesty: I cannot generate clinical advice or provide information beyond the provided text with clinical certainty.
    3. Next Step: Please consult a licensed medical professional or refer to appropriate external guidelines for safe and accurate guidance."
"""

# ==========================================
# 4. Main Layout (Two Columns)
# ==========================================
st.title("Ask a Clinical Question")
st.caption("Answering only from: WHO Guideline for the Pharmacological Treatment of Hypertension in Adults (2021)")

# تقسيم الشاشة إلى عمودين (60% للشات، 40% للأدلة)
col_main, col_evidence = st.columns([6, 4])

# مساحة إدخال السؤال
query = st.chat_input("What is the recommended blood pressure threshold...?")

if query:
    with col_main:
        st.markdown(f"**Question:** {query}")
        
        with st.spinner("Searching guidelines and generating safe response..."):
            retrieval_result = safe_query(query, top_k=3)
            res_data = retrieval_result["retrieved_chunks"]
            
            # بناء الإجابة
            if not retrieval_result["in_scope"]:
                final_answer = (
                    "**1. Insufficiency:** The provided WHO hypertension guideline does not contain data or recommendations to address your specific question.\n\n"
                    "**2. Honesty:** I cannot generate clinical advice or provide information beyond the provided text with clinical certainty.\n\n"
                    "**3. Next Step:** Please consult a licensed medical professional or refer to appropriate external guidelines for safe and accurate guidance."
                )
                st.error("Out of Scope / Safe Refusal Triggered")
                st.markdown(final_answer)
            else:
                context_blocks = []
                for i in range(len(res_data["ids"][0])):
                    chunk_id = res_data["ids"][0][i]
                    text = res_data["documents"][0][i]
                    meta = res_data["metadatas"][0][i]
                    context_blocks.append(
                        f"Document: {meta['document_name']}\nSection: {meta['section_title']}\nPage: {meta['page_numbers']}\nChunk: {chunk_id}\nText: {text}\n"
                    )
                
                full_context = "\n\n".join(context_blocks)
                user_prompt = f"CONTEXT:\n{full_context}\n\nUSER QUERY: {query}"
                
                model = genai.GenerativeModel(model_name="models/gemini-3.6-flash", system_instruction=RAG_SYSTEM_PROMPT)
                response = model.generate_content(user_prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                
                st.success("High Confidence Answer Generated")
                st.markdown(response.text)

    # عرض الأدلة في العمود الجانبي (Evidence Panel)
    with col_evidence:
        st.markdown("### Retrieved Evidence (Top-3 Chunks)")
        if "ids" in res_data and len(res_data["ids"]) > 0:
            for i in range(len(res_data["ids"][0])):
                doc_text = res_data["documents"][0][i]
                meta = res_data["metadatas"][0][i]
                dist = res_data["distances"][0][i]
                sim_score = round(1 - dist, 2)
                
                # استخدام HTML لرسم كروت الأدلة بشكل مطابق للصورة
                st.markdown(f"""
                <div class="evidence-box">
                    <span class="score-badge">{sim_score}</span> <strong>{meta['section_title']} (p.{meta['page_numbers']})</strong><br><br>
                    {doc_text[:250]}...
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No relevant chunks retrieved.")
