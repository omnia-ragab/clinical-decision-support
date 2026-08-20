import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re

# ==========================================
# 1. Page Configuration & Custom CSS (Deep Sea Dark Mode)
# ==========================================
st.set_page_config(page_title="Chronic Diseases Clinical Support ᴸᴵᵀᴱ", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Global Dark Mode Deep Sea Theme */
    .stApp {
        background-color: #0d1b2a !important;
        color: #e0e1dd !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1b263b !important;
    }
    [data-testid="stSidebar"] * {
        color: #e0e1dd !important;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px; text-align: left; background-color: #415a77 !important;
        color: #e0e1dd !important; border: none; width: 100%; margin-bottom: 5px; font-weight: bold;
    }
    .stButton>button:hover { background-color: #778da9 !important; color: #0d1b2a !important; }
    
    /* Main Output Cards */
    .recommendation-box {
        background-color: #1b263b; padding: 20px; border-radius: 8px;
        border-top: 5px solid #778da9; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        color: #e0e1dd;
    }
    .refusal-box {
        background-color: #2b1b1b; border: 1px solid #778da9; padding: 20px;
        border-radius: 8px; color: #e0e1dd; margin-bottom: 15px;
    }
    .evidence-box {
        background-color: #1b263b; color: #e0e1dd; padding: 15px; border-radius: 8px;
        border-left: 6px solid #415a77; margin-bottom: 15px; font-size: 0.9em;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .score-badge {
        background-color: #415a77; color: #e0e1dd; padding: 3px 8px; border-radius: 12px;
        font-size: 0.8em; font-weight: bold;
    }
    .json-box {
        background-color: #0d1b2a; color: #778da9; padding: 15px; border-radius: 8px;
        font-family: monospace; font-size: 0.85em; overflow-x: auto;
        border: 1px solid #415a77;
    }
    h1, h2, h3, h4, h5 { color: #e0e1dd !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Session State Initialization
# ==========================================
if "page" not in st.session_state: st.session_state.page = "Ask Question"
if "history" not in st.session_state: st.session_state.history = []
if "top_k" not in st.session_state: st.session_state.top_k = 3
if "threshold" not in st.session_state: st.session_state.threshold = 0.45

# ==========================================
# 3. Sidebar Navigation & Awareness Section
# ==========================================
with st.sidebar:
    st.markdown("### 🌐 Chronic Diseases Support")
    st.markdown("---")
    if st.button("💬 Ask Clinical Question"): st.session_state.page = "Ask Question"
    if st.button("📖 Chronic Awareness Hub"): st.session_state.page = "Awareness"
    if st.button("🕒 Query History"): st.session_state.page = "History"
    if st.button("📚 Guidelines Sources"): st.session_state.page = "Sources"
    if st.button("ℹ️ About System"): st.session_state.page = "About"
    if st.button("⚙️ System Guardrails"): st.session_state.page = "Settings"
    
    st.markdown("---")
    st.markdown("**Hackathon Project:** AI Clinical Decision Support Lite")
    st.markdown("**Theme:** Deep Sea & Evidence-Grounded RAG")

# ==========================================
# 4. Local Model & DB Initialization
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_databases():
    client_hyper = chromadb.PersistentClient(path="./chroma_db_hypertension")
    col_hyper = client_hyper.get_collection(name="who_hypertension_guideline_v2_cosine")
    
    client_diab = chromadb.PersistentClient(path="./chroma_db_diabetes")
    col_diab = client_diab.get_collection(name="who_diabetes_guideline_cosine")
    
    client_asthma = chromadb.PersistentClient(path="./chroma_db_asthma")
    col_asthma = client_asthma.get_collection(name="nice_asthma_guideline_cosine")
    
    return {"Hypertension (WHO 2021)": col_hyper, "Diabetes (WHO 2018)": col_diab, "Asthma (NICE 2024)": col_asthma}

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
    best_distance = results["distances"][0][0] if results["distances"][0] else 1.0
    if best_distance > st.session_state.threshold: 
        return {"in_scope": False, "retrieved_chunks": results, "best_distance": best_distance}
    return {"in_scope": True, "retrieved_chunks": results, "best_distance": best_distance}

def generate_local_response(res_data, guideline_name):
    top_doc = res_data["documents"][0][0]
    top_meta = res_data["metadatas"][0][0]
    
    clean_text = top_doc.replace("Title:", "").strip()
    if len(clean_text) > 400:
        clean_text = clean_text[:400] + "..."
        
    pages_fmt = top_meta['page_numbers'].replace("['", "").replace("']", "").replace("'", "")
    
    structured_data = {
        "recommendation": f"Based on {guideline_name}, key guidance retrieved from section '{top_meta['section_title']}': {clean_text[:200]}...",
        "evidence": clean_text,
        "citations": [
            {
                "document": top_meta['document_name'],
                "section": top_meta['section_title'],
                "page": pages_fmt
            }
        ],
        "confidence": "High"
    }
    return structured_data

# ==========================================
# 5. Page Routing & UI Rendering
# ==========================================
if st.session_state.page == "About":
    st.title("ℹ️ About Chronic Diseases Clinical Support")
    st.markdown("---")
    st.write("This application is an offline-first, zero-latency clinical decision support RAG pipeline built for managing chronic illnesses securely.")
    st.markdown("### Core Philosophy")
    st.info("Fluent $\\rightarrow$ Safe. All outputs are strictly grounded in verified public health guidelines with explicit page citations and zero hallucination guardrails.")

elif st.session_state.page == "Awareness":
    st.title("📖 Chronic Diseases Awareness & Education Hub")
    st.markdown("---")
    st.markdown("A comprehensive guide on the three major chronic conditions covered in our clinical database:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🩺 Hypertension")
        st.markdown("**Overview:** Chronic high blood pressure forcing blood against artery walls. Managed through pharmacological interventions (WHO 2021) and lifestyle modifications.")
    with col2:
        st.subheader("🩸 Diabetes Mellitus")
        st.markdown("**Overview:** Metabolic disorder characterized by elevated blood glucose levels due to insulin resistance or deficiency, requiring targeted treatments (WHO 2018).")
    with col3:
        st.subheader("🌬️ Asthma Management")
        st.markdown("**Overview:** Chronic inflammatory respiratory disease causing airway narrowing, monitored and managed via evidence-based protocols (NICE 2024).")

elif st.session_state.page == "Sources":
    st.title("📚 Official Guideline Sources")
    st.markdown("---")
    st.markdown("1. **WHO Guideline for the Pharmacological Treatment of Hypertension in Adults (2021)**")
    st.markdown("2. **WHO Guideline for Second- and Third-Line Medicines and Type of Insulin for Diabetes (2018)**")
    st.markdown("3. **NICE Guideline on Asthma: Diagnosis, Monitoring and Chronic Management (NG245 - 2024)**")

elif st.session_state.page == "Settings":
    st.title("⚙️ System Guardrails & Hyperparameters")
    st.markdown("---")
    st.session_state.top_k = st.slider("Top-K Chunks", 1, 5, st.session_state.top_k)
    st.session_state.threshold = st.slider("Distance Threshold (Guardrail)", 0.10, 0.60, st.session_state.threshold, 0.05)

elif st.session_state.page == "History":
    st.title("🕒 Query History")
    st.markdown("---")
    if not st.session_state.history:
        st.info("No queries recorded yet.")
    else:
        for item in reversed(st.session_state.history): 
            st.markdown(f"**Question ({item['guideline']}):** {item['question']}")
            st.write(item['answer'])
            st.markdown("---")

elif st.session_state.page == "Ask Question":
    st.title("Chronic Diseases Clinical Decision Support")
    st.markdown("Evidence-based guidance retrieval system for chronic disease management.")
    
    st.markdown("### 1. Select Clinical Domain")
    selected_guideline = st.radio("Choose the guideline context:", list(collections.keys()), horizontal=True)
    active_collection = collections[selected_guideline]
    
    # عرض صورة توضيحية حسب المرض المختار لتعزيز الـ Mood
    if "Hypertension" in selected_guideline:
        st.info("🩺 **Active Focus:** Hypertension (Blood Pressure Management & Guidelines)")
    elif "Diabetes" in selected_guideline:
        st.info("🩸 **Active Focus:** Diabetes Mellitus (Glucose & Insulin Management)")
    elif "Asthma" in selected_guideline:
        st.info("🌬️ **Active Focus:** Asthma (Respiratory Care & Diagnostics)")

    st.markdown("### 2. Enter Clinical Query")
    query = st.chat_input(f"Ask about {selected_guideline}...")

    if query:
        col_main, col_evidence = st.columns([6, 4])
        
        with col_main:
            st.markdown(f"**Question:** {query}")
            with st.spinner("Searching local vector database..."):
                retrieval_result = safe_query(active_collection, query, top_k=st.session_state.top_k)
                res_data = retrieval_result["retrieved_chunks"]
                best_dist = retrieval_result["best_distance"]
                
                if not retrieval_result["in_scope"]:
                    st.markdown(f"""
                    <div class="refusal-box">
                        <h3 style="color: #e0e1dd;">Refusal Example (Out-of-Scope Question)</h3>
                        <p><strong>I couldn't find enough information in the indexed guidelines to answer this confidently.</strong></p>
                        <p>This source doesn't appear to cover this topic. Please try rephrasing your question or consult a clinician directly.</p>
                        <hr style="border-color: #778da9;">
                        <p><strong>Why Refused?</strong></p>
                        <ul>
                            <li>No relevant chunks retrieved (similarity distance {round(best_dist, 2)} > threshold {st.session_state.threshold})</li>
                            <li>Question is outside the scope of the {selected_guideline} guideline</li>
                            <li>Zero hallucination policy enforced</li>
                        </ul>
                        <p><strong>Top similarity:</strong> {round(1 - best_dist, 2)} (below confidence threshold)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    structured_data = generate_local_response(res_data, selected_guideline)
                    
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h4 style="color:#778da9;">Recommendation</h4>
                        <p>{structured_data.get('recommendation', '')}</p>
                        <h5 style="color:#e0e1dd;">Evidence (Excerpt)</h5>
                        <p><i>"{structured_data.get('evidence', '')}"</i></p>
                        <p><strong>Confidence:</strong> {structured_data.get('confidence', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Structured Output (JSON)")
                    st.markdown(f'<div class="json-box">{json.dumps(structured_data, indent=2)}</div>', unsafe_allow_html=True)
                    
                    st.session_state.history.append({"question": query, "answer": structured_data.get('recommendation', ''), "guideline": selected_guideline})

        with col_evidence:
            st.markdown("### Retrieved Evidence (Top Chunks)")
            if res_data and "ids" in res_data and len(res_data["ids"][0]) > 0:
                for m, t, d in zip(res_data["metadatas"][0], res_data["documents"][0], res_data["distances"][0]):
                    pages_formatted = m['page_numbers'].replace("['", "").replace("']", "").replace("'", "")
                    st.markdown(f"""
                    <div class="evidence-box">
                        <span class="score-badge">Similarity: {round(1-d, 2)}</span><br>
                        <strong>Section: {m['section_title']}</strong><br>
                        <span style="color: #778da9; font-weight: bold;">Page(s): {pages_formatted}</span><br><br>
                        {t[:220]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No chunks retrieved.")
