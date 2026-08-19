import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import json
import re

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(page_title="Clinical Decision Support ᴸᴵᵀᴱ", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Deep Space Blue Sidebar */
    [data-testid="stSidebar"] { background-color: #003049 !important; }
    [data-testid="stSidebar"] * { color: #fdf0d5 !important; }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px; text-align: left; background-color: #669bbc !important;
        color: #fdf0d5 !important; border: none; width: 100%; margin-bottom: 5px;
    }
    .stButton>button:hover { background-color: #c1121f !important; color: white !important; }
    
    /* Main Output Cards */
    .recommendation-box {
        background-color: #ffffff; padding: 20px; border-radius: 8px;
        border-top: 4px solid #003049; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .evidence-box {
        background-color: #fdf0d5; color: #003049; padding: 15px; border-radius: 8px;
        border-left: 6px solid #669bbc; margin-bottom: 15px; font-size: 0.9em;
    }
    .score-badge {
        background-color: #669bbc; color: #fdf0d5; padding: 3px 8px; border-radius: 12px;
        font-size: 0.8em; font-weight: bold;
    }
    .json-box {
        background-color: #003049; color: #669bbc; padding: 15px; border-radius: 8px;
        font-family: monospace; font-size: 0.85em; overflow-x: auto;
    }
    /* Headers */
    h1, h2, h3 { color: #003049 !important; }
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
# 3. Sidebar Navigation
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ Clinical Decision Support ᴸᴵᵀᴱ")
    st.markdown("---")
    if st.button("💬 Ask Question"): st.session_state.page = "Ask Question"
    if st.button("🕒 History"): st.session_state.page = "History"
    if st.button("📚 Sources"): st.session_state.page = "Sources"
    if st.button("⚙️ Settings"): st.session_state.page = "Settings"

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
        return {"in_scope": False, "retrieved_chunks": results}
    return {"in_scope": True, "retrieved_chunks": results}

RAG_SYSTEM_PROMPT = """
You are a clinical decision support AI. Your ONLY source of truth is the provided clinical guidelines context.
Output your response STRICTLY as a valid JSON object with the following structure:
{
  "recommendation": "Concise recommendation based on context.",
  "evidence": "Supporting evidence excerpt from context.",
  "citations": [
    {"document": "Doc Name", "section": "Section Name", "page": "Page Num"}
  ],
  "confidence": "High/Medium/Low"
}
If out-of-scope or no context applies, output:
{
  "recommendation": "Out of Scope",
  "evidence": "Insufficient information in the provided guidelines.",
  "citations": [],
  "confidence": "Low"
}
"""

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else "{}"

# ==========================================
# 5. Page Routing
# ==========================================
if st.session_state.page == "Sources":
    st.title("📚 Official Guideline Sources")
    st.write("- **Hypertension (WHO 2021)**\n- **Diabetes (WHO 2018)**\n- **Asthma (NICE 2024)**")

elif st.session_state.page == "Settings":
    st.title("⚙️ System Guardrails")
    st.session_state.top_k = st.slider("Top-K Chunks", 1, 5, st.session_state.top_k)
    st.session_state.threshold = st.slider("Distance Threshold", 0.10, 0.60, st.session_state.threshold, 0.05)

elif st.session_state.page == "History":
    st.title("🕒 Query History")
    if not st.session_state.history:
        st.info("No queries recorded yet.")
    else:
        for item in reversed(st.session_state.history): 
            st.markdown(f"**Question ({item['guideline']}):** {item['question']}")
            st.write(item['answer'])
            st.markdown("---")

elif st.session_state.page == "Ask Question":
    st.title("Ask a Clinical Question")
    
    st.markdown("### 1. Select Clinical Domain")
    selected_guideline = st.radio("Choose the guideline context:", list(collections.keys()), horizontal=True)
    active_collection = collections[selected_guideline]
    
    st.markdown("### 2. Enter your Query")
    query = st.chat_input(f"Ask about {selected_guideline}...")

    if query:
        col_main, col_evidence = st.columns([6, 4])
        
        with col_main:
            st.markdown(f"**Question:** {query}")
            with st.spinner("Analyzing guidelines..."):
                retrieval_result = safe_query(active_collection, query, top_k=st.session_state.top_k)
                res_data = retrieval_result["retrieved_chunks"]
                
                if not retrieval_result["in_scope"]:
                    st.error("⚠️ Refusal Triggered: Out-of-Scope Question")
                else:
                    context_blocks = [f"Doc: {m['document_name']}\nSec: {m['section_title']}\nPage: {m['page_numbers']}\nText: {t}" for m, t in zip(res_data["metadatas"][0], res_data["documents"][0])]
                    user_prompt = f"CONTEXT:\n{chr(10).join(context_blocks)}\n\nUSER QUERY: {query}"
                    
                    try:
                        # استخدام اسم الموديل المستقر المحدث
                        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash", system_instruction=RAG_SYSTEM_PROMPT)
                        response = model.generate_content(user_prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                        
                        json_str = extract_json(response.text)
                        structured_data = json.loads(json_str)
                        
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h4 style="color:#c1121f;">Recommendation</h4>
                            <p>{structured_data.get('recommendation', '')}</p>
                            <h5 style="color:#669bbc;">Evidence (Excerpt)</h5>
                            <p><i>"{structured_data.get('evidence', '')}"</i></p>
                            <p><strong>Confidence:</strong> {structured_data.get('confidence', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### Structured Output (JSON)")
                        st.markdown(f'<div class="json-box">{json.dumps(structured_data, indent=2)}</div>', unsafe_allow_html=True)
                        
                        st.session_state.history.append({"question": query, "answer": structured_data.get('recommendation', ''), "guideline": selected_guideline})
                        
                    except json.JSONDecodeError:
                        st.error("Failed to parse structured JSON. Raw output:")
                        st.write(response.text)
                    except Exception as e:
                        err_msg = "⚠️ API Quota Exceeded. Please wait a minute and try again." if "429" in str(e) or "ResourceExhausted" in str(e) else f"Error: {e}"
                        st.warning(err_msg)

        with col_evidence:
            st.markdown("### Retrieved Evidence (Top Chunks)")
            if res_data and "ids" in res_data and len(res_data["ids"][0]) > 0:
                for m, t, d in zip(res_data["metadatas"][0], res_data["documents"][0], res_data["distances"][0]):
                    st.markdown(f"""
                    <div class="evidence-box">
                        <span class="score-badge">{round(1-d, 2)}</span> 
                        <strong>{m['section_title']} (p.{m['page_numbers']})</strong><br><br>
                        {t[:200]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No chunks retrieved.")
