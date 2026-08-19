import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np

# ==========================================
# 1. Page Configuration & Custom CSS (UI/UX)
# ==========================================
st.set_page_config(page_title="Medical RAG - WHO Hypertension", layout="wide", initial_sidebar_state="expanded")

# تطبيق الألوان (Earthy Muted Palette) مع بوكس السؤال
st.markdown("""
<style>
    /* Charcoal Blue Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2f3e46 !important;
    }
    /* Ash Grey Sidebar Text */
    [data-testid="stSidebar"] * {
        color: #cad2c5 !important;
    }
    /* Dark Slate Grey Buttons */
    .stButton>button {
        border-radius: 8px;
        text-align: left;
        background-color: #354f52 !important;
        color: #cad2c5 !important;
        border: none;
        width: 100%;
        margin-bottom: 5px;
    }
    /* Deep Teal Hover Effect */
    .stButton>button:hover {
        background-color: #52796f !important;
        color: white !important;
    }
    /* Ash Grey Evidence Box with Deep Teal Border */
    .evidence-box {
        background-color: #cad2c5;
        color: #2f3e46;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #52796f;
        margin-bottom: 15px;
        font-size: 0.9em;
    }
    /* Muted Teal Score Badge */
    .score-badge {
        background-color: #84a98c;
        color: #2f3e46;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    /* Custom Question Box */
    .question-box {
        background-color: #cad2c5; /* نفس لون الـ Ash Grey ليكون متناسق */
        color: #2f3e46; /* لون الكتابة غامق للوضوح */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #52796f; /* حدود بلون Deep Teal */
        margin-bottom: 20px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Session State Initialization
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "Ask Question"
if "history" not in st.session_state:
    st.session_state.history = []
if "top_k" not in st.session_state:
    st.session_state.top_k = 3
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.35

# متغيرات لحفظ السؤال الحالي علشان ميتسمحش لو غيرنا الصفحة
if "current_q" not in st.session_state:
    st.session_state.current_q = None
if "current_a" not in st.session_state:
    st.session_state.current_a = None
if "current_evidence" not in st.session_state:
    st.session_state.current_evidence = None

# ==========================================
# 3. Sidebar Navigation
# ==========================================
with st.sidebar:
    st.markdown("### 🩺 Medical RAG System")
    st.markdown("---")
    if st.button("💬 Ask Question"):
        st.session_state.page = "Ask Question"
    if st.button("🕒 History"):
        st.session_state.page = "History"
    if st.button("📚 Sources"):
        st.session_state.page = "Sources"
    if st.button("⚙️ Settings"):
        st.session_state.page = "Settings"

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
def load_chroma_collection():
    client = chromadb.PersistentClient(path=".")
    return client.get_collection(name="who_hypertension_guideline_v2_cosine")

embedding_model = load_embedding_model()
collection = load_chroma_collection()

def normalize_embedding(emb):
    vec = np.array(emb)
    norm = np.linalg.norm(vec)
    if norm > 0:
        return (vec / norm).tolist()
    return vec.tolist()

def safe_query(question, top_k, threshold):
    raw_emb = embedding_model.encode([question])[0]
    query_embedding = normalize_embedding(raw_emb)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    best_distance = results["distances"][0][0] if results["distances"][0] else 1.0
    
    if best_distance > threshold: 
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
# 5. Page Routing & Main Layout
# ==========================================

if st.session_state.page == "Sources":
    st.title("📚 Official Guideline Sources")
    st.markdown("---")
    st.markdown("### 1. WHO Guideline for the Pharmacological Treatment of Hypertension in Adults (2021)")
    st.write("**Published By:** World Health Organization")
    st.write("**Document Type:** Clinical Public Health Guidance")
    st.write("**Scope:** Pharmacological treatment of hypertension in non-pregnant adults.")
    st.info("All answers generated by this AI system are exclusively grounded in the text of this specific, legally usable PDF document. No parametric memory or external web search is utilized.")

elif st.session_state.page == "Settings":
    st.title("⚙️ System Guardrails & Parameters")
    st.markdown("---")
    st.write("Adjust the RAG pipeline parameters to control retrieval and safety thresholds.")
    
    st.session_state.top_k = st.slider(
        "Top-K Chunks (Number of text chunks retrieved)", 
        min_value=1, max_value=5, value=st.session_state.top_k, step=1
    )
    
    st.session_state.threshold = st.slider(
        "Cosine Distance Threshold (Lower means stricter safety refusal)", 
        min_value=0.10, max_value=0.60, value=st.session_state.threshold, step=0.05
    )
    st.success("Settings saved automatically!")

elif st.session_state.page == "History":
    st.title("🕒 Query History")
    st.markdown("---")
    if not st.session_state.history:
        st.info("No questions asked yet. Go to 'Ask Question' to start.")
    else:
        for item in reversed(st.session_state.history): 
            st.markdown(f"""
            <div class="question-box">
                <strong>Question:</strong> "{item['question']}"
            </div>
            """, unsafe_allow_html=True)
            st.markdown(item['answer'])
            st.markdown("---")

elif st.session_state.page == "Ask Question":
    st.title("🩺 Medical RAG System — WHO Hypertension")
    st.caption("Answering only from: WHO Guideline for the Pharmacological Treatment of Hypertension in Adults (2021)")

    # تقسيم الشاشة لعمودين (السؤال/الإجابة 60% - الأدلة 40%)
    col_main, col_evidence = st.columns([6, 4])

    query = st.chat_input("What is the recommended blood pressure threshold...?")

    if query:
        st.session_state.current_q = query
        
        with col_main:
            with st.spinner("Searching guidelines and generating safe response..."):
                retrieval_result = safe_query(query, top_k=st.session_state.top_k, threshold=st.session_state.threshold)
                res_data = retrieval_result["retrieved_chunks"]
                
                st.session_state.current_evidence = res_data
                
                if not retrieval_result["in_scope"]:
                    final_answer = (
                        "**1. Insufficiency:** The provided WHO hypertension guideline does not contain data or recommendations to address your specific question.\n\n"
                        "**2. Honesty:** I cannot generate clinical advice or provide information beyond the provided text with clinical certainty.\n\n"
                        "**3. Next Step:** Please consult a licensed medical professional or refer to appropriate external guidelines for safe and accurate guidance."
                    )
                    st.error("Out of Scope / Safe Refusal Triggered")
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
                    final_answer = response.text
                
                st.session_state.current_a = final_answer
                
                st.session_state.history.append({
                    "question": query,
                    "answer": final_answer
                })

    # عرض السؤال الحالي وإجابته
    if st.session_state.current_q:
        with col_main:
            st.markdown(f"""
            <div class="question-box">
                <strong>Question:</strong> "{st.session_state.current_q}"
            </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.current_a)

        # عرض الأدلة في العمود الجانبي بألوان تناسب الـ Palette
        with col_evidence:
            st.markdown("### Retrieved Evidence (Top Chunks)")
            res_data = st.session_state.current_evidence
            if res_data and "ids" in res_data and len(res_data["ids"][0]) > 0:
                for i in range(len(res_data["ids"][0])):
                    doc_text = res_data["documents"][0][i]
                    meta = res_data["metadatas"][0][i]
                    dist = res_data["distances"][0][i]
                    sim_score = round(1 - dist, 2)
                    
                    st.markdown(f"""
                    <div class="evidence-box">
                        <span class="score-badge">{sim_score}</span> <strong>{meta['section_title']} (p.{meta['page_numbers']})</strong><br><br>
                        {doc_text[:250]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No relevant chunks retrieved.")
