# 🛡️ PolicySarthi: Advanced AI Policy Assistant

**PolicySarthi** is a next-generation, multi-modal assistant designed to revolutionize how hospital staff and administrators access complex policy documents and insurance claim requirements. Built for the **Sarvam AI Hackathon**, it combines the best of Retrieval-Augmented Generation (RAG) with a modern self-learning loop.

---

## 🚀 Key Features

### 📄 1. High-Fidelity RAG (Multi-Format)
The system doesn't just read JSON; it utilizes **FAISS** to semantically index and search across both structured data and complex **PDF documents** (e.g., standard hospital SOPs and insurance policy booklets).
- **200+ Document Chunks** indexed for deep semantic retrieval.
- **Context-Grounded Answers:** Assistant never "hallucinates"; it cites its sources directly.

### 🎙️ 2. Multilingual Voice Integration (STT & TTS)
Powered by **Sarvam AI**, PolicySarthi supports natural voice interactions.
- **Voice Messages:** Record queries in Hindi or English and get instant transcriptions.
- **Audio Playback:** The assistant can speak back its findings using high-quality Indian-language TTS.

### 📷 3. Visual Intelligence (OCR/Vision)
Upload images of medical bills, insurance cards, or printed policy notices.
- **Sarvam Vision:** Extracts high-accuracy text from images and PDF scans.
- **Automated Analysis:** If you upload a medical bill, the assistant will automatically analyze it against the policy corpus to check for coverage details.

### 🧠 4. Dynamic Self-Learning Core
The project features a **Continuous Improvement Loop**.
- **User Feedback:** Every answer can be rated and corrected.
- **Adaptive Memory:** The system "learns" from corrections in real-time. Future queries will automatically apply these "Lessons Learned" to prioritize user-verified facts.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **LLM & AI APIs** | Sarvam AI (Saarika, Saaras, Document Intel) |
| **Vector Database** | FAISS (Meta AI) |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) |
| **Backend Framework** | FastAPI (Python 3.11+) |
| **Frontend UI** | Streamlit |
| **Local Storage** | SQLite (User Logs, Feedback & Lessons) |

---

## 🏁 Quick Start Guide

### 1. Prerequites
Ensure you have Python 3.11+ and a virtual environment activated:
```powershell
.\.venv\Scripts\activate
```

### 2. Run the Backend (FastAPI)
The backend handles indexing, OCR, and the RAG pipeline.
```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 5000
```

### 3. Run the Frontend (Streamlit)
The frontend provides the chat, voice, and visual interface.
```powershell
streamlit run frontend\main.py
```

### 4. Presentation Credentials
| Role | Username | Password |
| :--- | :--- | :--- |
| **Admin** | `admin` | `admin123` |
| **Staff** | `staff` | `staff123` |

---

## 💡 Hackathon Demo Instructions

To showcase the **Self-Learning Loop**:
1. Ask: *"What is the policy for Ayushman Bharat?"*
2. Give a **Correction** via the feedback icon: *"Ayushman Bharat covers up to 5 Lakhs per family per year."*
3. Ask the question again.
4. Observe the assistant display the ✨ **"Learning Applied"** badge and use your specific correction!

---
**Developed for the Advanced Agentic Coding Hackathon.**
*PolicySarthi — Your Intelligent Insurance Companion.*
