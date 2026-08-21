import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import google.generativeai as genai

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(page_title="SoloRAG Clinical Support", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0d1b2a !important; color: #e0e1dd !important; }
    [data-testid="stSidebar"] { background-color: #1b263b !important; }
    [data-testid="stSidebar"] * { color: #e0e1dd !important; }
    .stButton>button { border-radius: 8px; text-align: left; background-color: #415a77 !important; color: #e0e1dd !important; border: none; width: 100%; margin-bottom: 5px; font-weight: bold; }
    .stButton>button:hover { background-color: #778da9 !important; color: #0d1b2a !important; }
    .recommendation-box { background-color: #1b263b; padding: 20px; border-radius: 8px; border-top: 5px solid #778da9; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); color: #e0e1dd; }
    .refusal-box { background-color: #2b1b1b; border: 1px solid #778da9; padding: 20px; border-radius: 8px; color: #e0e1dd; margin-bottom: 15px; }
    .evidence-box { background-color: #1b263b; color: #e0e1dd; padding: 15px; border-radius: 8px; border-left: 6px solid #415a77; margin-bottom: 15px; font-size: 0.9em; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
    .score-badge { background-color: #415a77; color: #e0e1dd; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
    h1, h2, h3, h4, h5 { color: #e0e1dd !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Gemini API Configuration (With Safety)
# ==========================================
API_AVAILABLE = False
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        API_AVAILABLE = True
except Exception:
    pass

RAG_SYSTEM_PROMPT = """
You are a clinical decision support AI acting as an evidence synthesizer.
Your ONLY source of truth is the provided clinical guidelines context.
CORE PHILOSOPHY: Fluent -> Safe.
RULES:
1. If the context contains the answer, summarize it concisely under "Recommendation", followed by "Supporting Evidence".
2. CITATIONS ARE MANDATORY: Every claim must end with a citation strictly in this format: [Document Name - Section: <Section Name> - Page <Page Number>].
3. Never act as a diagnostician. Do not use your parametric memory.
"""

# ==========================================
# 3. Session State Initialization
# ==========================================
if "page" not in st.session_state: st.session_state.page = "Ask Question"
if "history" not in st.session_state: st.session_state.history = []
if "top_k" not in st.session_state: st.session_state.top_k = 3
if "threshold" not in st.session_state: st.session_state.threshold = 0.45

# ==========================================
# 4. Sidebar Navigation
# ==========================================
with st.sidebar:
    st.markdown("### SoloRAG Support")
    st.markdown("---")
    if st.button("Ask Clinical Question"): st.session_state.page = "Ask Question"
    if st.button("Chronic Awareness Hub"): st.session_state.page = "Awareness"
    if st.button("About System"): st.session_state.page = "About"
    if st.button("Query History"): st.session_state.page = "History"
    if st.button("System Guardrails"): st.session_state.page = "Settings"
    
    # --- Footer Added Here ---
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #778da9;'>✨ Created by <strong>Omnia Ragab</strong></p>", unsafe_allow_html=True)

# ==========================================
# 5. Local Model & DB Initialization (ALL 3 DISEASES)
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_databases():
    databases = {}
    try:
        col_hyper = chromadb.PersistentClient(path="./chroma_db_hypertension").get_or_create_collection(name="who_hypertension_guideline_v2_cosine")
        databases["Hypertension (WHO 2021)"] = col_hyper
    except: pass

    try:
        col_diab = chromadb.PersistentClient(path="./chroma_db_diabetes").get_or_create_collection(name="who_diabetes_guideline_cosine")
        databases["Diabetes (WHO 2018)"] = col_diab
    except: pass

    try:
        col_asthma = chromadb.PersistentClient(path="./chroma_db_asthma").get_or_create_collection(name="nice_asthma_guideline_cosine")
        databases["Asthma (NICE 2024)"] = col_asthma
    except: pass

    return databases

embedding_model = load_embedding_model()
collections = load_databases()

def normalize_embedding(emb):
    vec = np.array(emb)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()

def safe_query(collection, question, top_k=3):
    raw_emb = embedding_model.encode([question])[0]
    query_embedding = normalize_embedding(raw_emb)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    best_distance = results["distances"][0][0] if results["distances"] and results["distances"][0] else 1.0
    if best_distance > st.session_state.threshold: 
        return {"in_scope": False, "retrieved_chunks": results, "best_distance": best_distance}
    return {"in_scope": True, "retrieved_chunks": results, "best_distance": best_distance}

# ==========================================
# ROBUST GENERATION WITH LOCAL FALLBACK
# ==========================================
def generate_robust_response(res_data, query, guideline_name):
    if API_AVAILABLE:
        try:
            context_blocks = []
            for i in range(len(res_data["ids"][0])):
                text = res_data["documents"][0][i]
                meta = res_data["metadatas"][0][i]
                pages_fmt = meta['page_numbers'].replace("['", "").replace("']", "").replace("'", "")
                context_blocks.append(f"Document: {meta['document_name']} | Section: {meta['section_title']} | Page: {pages_fmt}\nText: {text}\n")
            
            full_context = "\n".join(context_blocks)
            user_prompt = f"CONTEXT:\n{full_context}\n\nUSER QUERY: {query}"
            
            model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=RAG_SYSTEM_PROMPT)
            response = model.generate_content(user_prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            return response.text
        except Exception:
            pass 
            
    top_doc = res_data["documents"][0][0]
    top_meta = res_data["metadatas"][0][0]
    clean_text = top_doc.replace("Title:", "").strip()
    if len(clean_text) > 500: clean_text = clean_text[:500] + "..."
    pages_fmt = top_meta['page_numbers'].replace("['", "").replace("']", "").replace("'", "")
    
    fallback_response = f"""
**Recommendation:**
Based on the {guideline_name}, key guidance retrieved from section '{top_meta['section_title']}':
{clean_text[:250]}...

**Supporting Evidence:**
"{clean_text}"

*[{top_meta['document_name']} - Section: {top_meta['section_title']} - Page {pages_fmt}]*
*(Note: System utilizing local offline fallback synthesis mode for uninterrupted access).*
    """
    return fallback_response

# ==========================================
# 6. Page Routing & UI Rendering
# ==========================================
if st.session_state.page == "About":
    st.title("About Chronic Diseases Clinical Support")
    st.markdown("---")
    st.write("This application is an offline-first, zero-latency clinical decision support RAG pipeline built for managing chronic illnesses securely.")
    st.markdown("### Core Philosophy")
    st.info("Fluent -> Safe. All outputs are strictly grounded in verified public health guidelines with explicit page citations and zero hallucination guardrails.")

elif st.session_state.page == "Awareness":
    st.title("Chronic Diseases Awareness Hub")
    selected_disease = st.selectbox("Choose a disease:", ["Hypertension (High Blood Pressure)", "Diabetes Mellitus", "Asthma Management"])
    
    st.subheader(f"{selected_disease} Clinical Overview")
    if "Hypertension" in selected_disease:
        st.write("According to WHO guidelines, pharmacological treatment should be initiated for individuals with confirmed hypertension when systolic blood pressure is >=140 mmHg or diastolic is >=90 mmHg.")
    elif "Diabetes" in selected_disease:
        st.write("WHO guidelines emphasize targeted second- and third-line medication management, proper insulin selection, and strict monitoring protocols.")
    elif "Asthma" in selected_disease:
        st.write("NICE guidelines highlight objective diagnostic testing and recommend modern management strategies including low-dose ICS combined with formoterol.")

elif st.session_state.page == "Settings":
    st.title("System Guardrails")
    st.session_state.top_k = st.slider("Top-K Chunks", 1, 5, st.session_state.top_k)
    st.session_state.threshold = st.slider("Distance Threshold (Guardrail)", 0.10, 0.60, st.session_state.threshold, 0.05)

elif st.session_state.page == "History":
    st.title("Query History")
    if not st.session_state.history: st.info("No queries recorded yet.")
    for item in reversed(st.session_state.history): 
        st.markdown(f"**Q:** {item['question']}")
        st.info(item['answer'])

elif st.session_state.page == "Ask Question":
    st.title("SoloRAG: Clinical Decision Support")
    
    if not collections:
        st.error("No databases loaded. Please check your chroma_db folders.")
    else:
        selected_guideline = st.selectbox("Choose context:", list(collections.keys()))
        active_collection = collections[selected_guideline]
        
        query = st.chat_input("Ask a clinical question...")

        if query:
            col_main, col_evidence = st.columns([6, 4])
            
            with col_main:
                st.markdown(f"**Question:** {query}")
                with st.spinner("Synthesizing evidence..."):
                    retrieval_result = safe_query(active_collection, query, top_k=st.session_state.top_k)
                    res_data = retrieval_result["retrieved_chunks"]
                    best_dist = retrieval_result["best_distance"]
                    
                    if not retrieval_result["in_scope"]:
                        st.markdown(f"""
                        <div class="refusal-box">
                            <h3 style="color: #e0e1dd;">🛡️ Clinical Safety Guardrail Triggered (Safe Refusal)</h3>
                            <p><strong>1. Insufficiency:</strong> The indexed public health guidelines do not contain data or recommendations covering emergency individual interventions, severe acute symptoms, or personalized prescriptions.</p>
                            <p><strong>2. Honesty:</strong> As a clinical decision support tool, I cannot generate specific medical advice or handle out-of-scope emergency queries beyond our validated document scope.</p>
                            <p><strong>3. Next Step:</strong> Please consult a licensed medical professional immediately or contact emergency services for proper acute clinical management.</p>
                            <hr style="border-color: #778da9;">
                            <p><strong>Guardrail Metrics:</strong> Distance {round(best_dist, 2)} > Threshold {st.session_state.threshold} (Out of Scope)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        final_answer = generate_robust_response(res_data, query, selected_guideline)
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h4 style="color:#778da9;">Synthesized Recommendation</h4>
                            <p>{final_answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.history.append({"question": query, "answer": final_answer})

            with col_evidence:
                st.markdown("### Retrieved Evidence (Top Chunks)")
                if res_data and "ids" in res_data and len(res_data["ids"][0]) > 0:
                    for m, t, d in zip(res_data["metadatas"][0], res_data["documents"][0], res_data["distances"][0]):
                        pages_fmt = m['page_numbers'].replace("['", "").replace("']", "").replace("'", "")
                        st.markdown(f"""
                        <div class="evidence-box">
                            <span class="score-badge">Similarity: {round(1-d, 2)}</span><br>
                            <strong>{m['section_title']}</strong> (Page: {pages_fmt})<br><br>
                            {t[:200]}...
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No chunks retrieved.")
