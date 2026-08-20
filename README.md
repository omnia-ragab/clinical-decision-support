# Chronic Diseases Clinical Decision Support ᴸᴵᵀᴱ

An offline-retrieval, AI-powered Clinical Decision Support system engineered for secure, evidence-based chronic disease management. Built for the AI Clinical Decision Support Hackathon (Organized by ITIDA, TIEC, Orange Digital Center, and Creativa)[cite: 2].

---

## 🛡️ Core Philosophy
**Fluent $\rightarrow$ Safe[cite: 2].** Clinical decision support must be strictly grounded in official evidence with explicit page-level citations, transparent retrieval, and verified safe refusal logic powered by state-of-the-art generative AI.

---

## 📚 Covered Clinical Guidelines
The system indexes and queries three official, public health guideline databases:
1. **Hypertension:** WHO Guideline for the Pharmacological Treatment of Hypertension in Adults (2021)[cite: 1, 2].
2. **Diabetes:** WHO Guideline for Second- and Third-Line Medicines and Type of Insulin for Diabetes (2018)[cite: 1, 2].
3. **Asthma:** NICE Guideline on Asthma: Diagnosis, Monitoring and Chronic Management (NG245 - 2024)[cite: 2].

---

## ⚙️ System Architecture & Tech Stack
* **Frontend UI:** Streamlit (Custom "Deep Sea" Dark Mode Theme).
* **Generative Engine:** Google Gemini API (`gemini-flash` / `gemini-1.5-flash`) with structured JSON outputs and robust fallback mechanisms[cite: 1, 2].
* **Vector Database:** ChromaDB (Persistent local storage with cosine similarity indexing)[cite: 1, 2].
* **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`) for local semantic vector representation[cite: 1, 2].
* **Guardrails & Safety:** Semantic distance thresholding with strict out-of-scope refusal logic[cite: 1, 2].

---

## 🚀 Key Features
* **Hybrid RAG Architecture:** Combines local vector search (ChromaDB + Sentence-Transformers) for accurate evidence retrieval with Gemini API for precise medical synthesis[cite: 1, 2].
* **Traceable Citations:** Every recommendation includes exact document names, section titles, and page-level citations[cite: 1, 2].
* **Structured JSON Outputs:** Generates standardized JSON objects containing recommendations, excerpts, confidence scores, and citations[cite: 1, 2].
* **Chronic Awareness Hub:** Interactive educational section featuring detailed clinical insights and guideline visualizations[cite: 1, 2].
* **Safe Refusal Mechanism:** Actively blocks unverified or out-of-scope queries to maintain high clinical safety standards[cite: 1, 2].

---

## 🛠️ Project Structure
```text
clinical-decision-support/
│
├── app.py                  # Main Streamlit application
├── chroma_db_hypertension/ # Persistent ChromaDB for Hypertension
├── chroma_db_diabetes/     # Persistent ChromaDB for Diabetes
├── chroma_db_asthma/       # Persistent ChromaDB for Asthma
├── image/                  # Clinical guidelines visuals & illustrations
│   ├── hypertension.jpg
│   ├── diabetes.jpg
│   └── asthma.jpg
└── README.md               # Project documentation
