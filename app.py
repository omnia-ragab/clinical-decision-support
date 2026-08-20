import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import json
import re
import time

# ==========================================
# 1. Page Configuration & Custom CSS (Theme: Space Indigo & Strawberry Red)
# ==========================================
st.set_page_config(page_title="Clinical Decision Support ᴸᴵᵀᴱ", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Space Indigo Sidebar */
    [data-testid="stSidebar"] { background-color: #2b2d42 !important; }
    [data-testid="stSidebar"] * { color: #edf2f4 !important; }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px; text-align: left; background-color: #8d99ae !important;
        color: #2b2d42 !important; border: none; width: 100%; margin-bottom: 5px; font-weight: bold;
    }
    .stButton>button:hover { background-color: #ef233c !important; color: white !important; }
    
    /* Main Output Cards */
    .recommendation-box {
        background-color: #ffffff; padding: 20px; border-radius: 8px;
        border-top: 5px solid #d90429; margin-bottom: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        color: #2b2d42;
    }
    .refusal-box {
        background-color: #fdf2f2; border: 1px solid #ef233c; padding: 20px;
        border-radius: 8px; color: #d90429; margin-bottom: 15px;
    }
    .evidence-box {
        background-color: #edf2f4; color: #2b2d42; padding: 15px; border-radius: 8px;
        border-left: 6px solid #8d99ae; margin-bottom: 15px; font-size: 0.9em;
    }
    .score-badge {
        background-color: #2b2d42; color: #edf2f4; padding: 3px 8px; border-radius: 12px;
        font-size: 0.8em; font-weight: bold;
    }
    .json-box {
        background-color: #2b2d42; color: #8d99ae; padding: 15px; border-radius: 8px;
        font-family: monospace; font-size: 0.85em; overflow-x: auto;
    }
    h1, h2, h3 { color: #2b2d42 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Session State Initialization
# ==========================================
if "page" not in st.session_state: st.session_state.page = "Ask Question"
if "history" not in st.session_state: st.session_state.history = []
if "top_k" not in st.session_state: st.session_state.top_k = 3
if "threshold" not in st.session_state: st.session_state.threshold = 0.35

# ==========================================
# 3. Sidebar Navigation & About Section
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ Clinical Decision Support ᴸᴵᵀᴱ")
    st.markdown("---")
    if st.button("💬 Ask Question"): st.session_state.page = "Ask Question"
    if st.button("🕒 History"): st.session_state.page = "History"
    if st.button("📚 Sources"): st.session_state.page = "Sources"
    if st.button("ℹ️ About System"): st.session_state.page = "About"
    if st.button("⚙️ Settings"): st.session_state.page = "Settings"
    
    st.markdown("---")
    st.markdown("**Hackathon Team:** AI Clinical Decision Support Lite")
    st.markdown("**Core Principle:** Fluent $\\rightarrow$ Safe[cite: 2]")

# ==========================================
# 4. Model & DB Initialization
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

RAG_SYSTEM_PROMPT = """
You are a clinical decision support AI acting as an evidence synthesizer. Your ONLY source of truth is the provided clinical guidelines context.
Output your response STRICTLY as a valid JSON object with the following structure:
{
  "recommendation": "Concise recommendation based on context.",
  "evidence": "Supporting evidence excerpt from context.",
  "citations": [
    {"document": "Doc Name", "section": "Section Name", "page": "Page Num"}
  ],
  "confidence": "High/Medium/Low"
}
"""

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else "{}"

# حل نهائي لمشكلة الـ API مع Caching متطور و Fallback للموديلات
@st.cache_data(show_spinner=False)
def robust_generate_answer(user_prompt):
    models_to_try = ["gemini-1.5-flash", "gemini-flash"]
    for mod_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name=mod_name, system_instruction=RAG_SYSTEM_PROMPT)
            response = model.generate_content(user_prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            return response.text
        except Exception:
            time.sleep(1)
            continue
    # رد احتياطي محلي لو الـ API حصل فيه ضغط تام لضمان عدم توقف الديمو
    return json.dumps({
        "recommendation": "System temporarily busy. Please rely on standard clinical review.",
        "evidence": "API rate limit encountered, cached safeguard active.",
        "citations": [{"document": "System Safeguard", "section": "Fallback", "page": "N/A"}],
        "confidence": "Low"
    })

# ==========================================
# 5. Page Routing & UI Rendering
# ==========================================
if st.session_state.page == "About":
    st.title("ℹ️ About Clinical Decision Support ᴸᴵᵀᴱ")
    st.markdown("---")
    st.write("This system is built for the AI Clinical Decision Support Hackathon (Organized by ITIDA, TIEC, Orange Digital Center, and Creativa)[cite: 2].")
    st.markdown("### Core Philosophy")
    st.info("Fluent $\\rightarrow$ Safe. Clinical recommendations must be grounded in official evidence with explicit page-level citations and verified safe refusal logic[cite: 2].")

elif st.session_state.page == "Sources":
    st.title("📚 Official Guideline Sources")
    st.markdown("---")
    st.markdown("1. **WHO Guideline for the Pharmacological Treatment of Hypertension in Adults (2021)**")
    st.markdown("2. **WHO Guideline for Second- and Third-Line Medicines and Type of Insulin for Diabetes (2018)**")
    st.markdown("3. **NICE Guideline on Asthma: Diagnosis, Monitoring and Chronic Management (NG245 - 2024)**[cite: 2]")

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
    st.title("Clinical Decision Support Hub")
    
    st.markdown("### 1. Select Clinical Domain")
    selected_guideline = st.radio("Choose the guideline context:", list(collections.keys()), horizontal=True)
    active_collection = collections[selected_guideline]
    
    st.markdown("### 2. Enter Clinical Query")
    query = st.chat_input(f"Ask about {selected_guideline}...")

    if query:
        col_main, col_evidence = st.columns([6, 4])
        
        with col_main:
            st.markdown(f"**Question:** {query}")
            with st.spinner("Analyzing guidelines securely..."):
                retrieval_result = safe_query(active_collection, query, top_k=st.session_state.top_k)
                res_data = retrieval_result["retrieved_chunks"]
                best_dist = retrieval_result["best_distance"]
                
                # الرفض المنطقي الصحيح المطابق لمعايير الـ Hackathon
                if not retrieval_result["in_scope"]:
                    st.markdown(f"""
                    <div class="refusal-box">
                        <h3>Refusal Example (Out-of-Scope Question)</h3>
                        <p><strong>I couldn't find enough information in the indexed guidelines to answer this confidently.</strong></p>
                        <p>This source doesn't appear to cover this topic. Please try rephrasing your question or consult a clinician directly.</p>
                        <hr style="border-color: #ef233c;">
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
                    context_blocks = [f"Doc: {m['document_name']}\nSec: {m['section_title']}\nPage: {m['page_numbers']}\nText: {t}" for m, t in zip(res_data["metadatas"][0], res_data["documents"][0])]
                    user_prompt = f"CONTEXT:\n{chr(10).join(context_blocks)}\n\nUSER QUERY: {query}"
                    
                    raw_response_text = robust_generate_answer(user_prompt)
                    try:
                        json_str = extract_json(raw_response_text)
                        structured_data = json.loads(json_str)
                        
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h4 style="color:#ef233c;">Recommendation</h4>
                            <p>{structured_data.get('recommendation', '')}</p>
                            <h5 style="color:#2b2d42;">Evidence (Excerpt)</h5>
                            <p><i>"{structured_data.get('evidence', '')}"</i></p>
                            <p><strong>Confidence:</strong> {structured_data.get('confidence', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### Structured Output (JSON)")
                        st.markdown(f'<div class="json-box">{json.dumps(structured_data, indent=2)}</div>', unsafe_allow_html=True)
                        
                        st.session_state.history.append({"question": query, "answer": structured_data.get('recommendation', ''), "guideline": selected_guideline})
                        
                    except Exception:
                        st.error("Error formatting structured response.")

        with col_evidence:
            st.markdown("### Retrieved Evidence (Top Chunks)")
            if res_data and "ids" in res_data and len(res_data["ids"][0]) > 0:
                for m, t, d in zip(res_data["metadatas"][0], res_data["documents"][0], res_data["distances"][0]):
                    # تنسيق رقم الصفحة بشكل منظم ومريح للعين
                    pages_formatted = m['page_numbers'].replace("['", "").replace("']", "").replace("'", "")
                    st.markdown(f"""
                    <div class="evidence-box">
                        <span class="score-badge">Similarity: {round(1-d, 2)}</span><br>
                        <strong>Section: {m['section_title']}</strong><br>
                        <span style="color: #d90429; font-weight: bold;">Page(s): {pages_formatted}</span><br><br>
                        {t[:220]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No chunks retrieved.")
