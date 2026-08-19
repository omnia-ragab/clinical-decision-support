import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Medical RAG - WHO Hypertension", page_icon="🩺")
st.title("🩺 Medical RAG System — WHO Hypertension")

# 2. تحميل API Key بتاع Gemini من إعدادات Streamlit
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
except KeyError:
    st.error("Please set GEMINI_API_KEY in Streamlit Secrets!")
    st.stop()

# 3. تحميل النماذج وقاعدة البيانات (بنستخدم st.cache_resource عشان تتحمل مرة واحدة بس)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_chroma_collection():
    # لازم فولدر chroma_db يكون موجود مع ملفات المشروع
    client = chromadb.PersistentClient(path=".")
    # تأكدي إن ده اسم الكولكشن النهائي بتاعك
    return client.get_collection(name="who_hypertension_guideline_v2_cosine")

embedding_model = load_embedding_model()
collection = load_chroma_collection()

# 4. دالة الـ Normalization (لأنك استخدمتي Cosine Similarity)
def normalize_embedding(emb):
    vec = np.array(emb)
    norm = np.linalg.norm(vec)
    if norm > 0:
        return (vec / norm).tolist()
    return vec.tolist()

# 5. دوال البحث من النوت بوك
def safe_query(question, top_k=3):
    raw_emb = embedding_model.encode([question])[0]
    query_embedding = normalize_embedding(raw_emb)
    
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    # حساب المسافة (التشابه)
    best_distance = results["distances"][0][0] if results["distances"][0] else 1.0
    
    # حد الأمان (Threshold) اللي يحدد هل السؤال داخل النطاق ولا لأ
    if best_distance > 0.35: 
        return {"in_scope": False}
        
    return {
        "in_scope": True,
        "retrieved_chunks": results
    }

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
4. Never act as a diagnostician. Do not use your parametric memory.
"""

def generate_grounded_answer(query):
    retrieval_result = safe_query(query, top_k=3)
    
    if not retrieval_result["in_scope"]:
        return (
            "1. Insufficiency: The provided WHO hypertension guideline does not contain data or recommendations to address your specific question.\n"
            "2. Honesty: I cannot generate clinical advice or provide information beyond the provided text with clinical certainty.\n"
            "3. Next Step: Please consult a licensed medical professional or refer to appropriate external guidelines for safe and accurate guidance."
        )
        
    context_blocks = []
    res_data = retrieval_result["retrieved_chunks"]
    for i in range(len(res_data["ids"][0])):
        chunk_id = res_data["ids"][0][i]
        text = res_data["documents"][0][i]
        meta = res_data["metadatas"][0][i]
        
        context_blocks.append(
            f"--- CHUNK ID: {chunk_id} ---\n"
            f"Document: {meta['document_name']}\n"
            f"Section: {meta['section_title']}\n"
            f"Page: {meta['page_numbers']}\n"
            f"Text: {text}\n"
        )
        
    full_context = "\n\n".join(context_blocks)
    user_prompt = f"CONTEXT:\n{full_context}\n\nUSER QUERY: {query}"
    
    # توليد الإجابة باستخدام Gemini
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", 
        system_instruction=RAG_SYSTEM_PROMPT
    )
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.0)
    )
    return response.text

# 6. واجهة المحادثة (Chat UI)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about WHO Hypertension Guidelines..."):
    # عرض سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # عرض إجابة الموديل
    with st.chat_message("assistant"):
        with st.spinner("Searching guidelines..."):
            answer = generate_grounded_answer(prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
