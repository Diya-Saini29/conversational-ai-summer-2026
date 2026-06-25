# 🗣️ Conversational AI — Summer 2026

**Building production-ready, voice-enabled conversational agents with RAG, LangChain, and Speech AI.**

> A journey from foundations to deployment — exploring transformers, retrieval-augmented generation, speech synthesis, and agentic workflows.

---

## 📌 What This Project Is About

Most AI demos stop at simple chatbots. This repository goes further — building **stateful, multi-turn conversational agents** that can:

- 🔍 **Retrieve knowledge** from documents (PDFs, text) using RAG
- 🎤 **Understand speech** via Whisper (STT)
- 🗣️ **Respond with voice** using TTS (SpeechT5 / Coqui)
- 🧠 **Remember context** across conversations using LangGraph checkpointers
- 🛠️ **Use tools** — search the web, call APIs, run Python functions
- 🚀 **Deploy** as a live web app with FastAPI + Streamlit

---

## 🎯 Why This Matters

| Traditional Chatbots | This Project |
|----------------------|--------------|
| Stateless (no memory) | Stateful memory across turns |
| Fixed responses | Dynamic, tool-using agents |
| Text-only | Voice + text multimodal |
| No document access | RAG-powered knowledge retrieval |
| Linear prompts | Cyclic graph-based workflows |

---

## 🗺️ 12-Week Learning Roadmap (Reflected in Code)

| Phase | Focus | Key Deliverable |
|-------|-------|-----------------|
| **Weeks 1-2** | Foundations — Transformers, Hugging Face, Fine-tuning | Notebooks on attention, tokenization |
| **Weeks 3-4** | RAG + LangChain + Vector DBs + Observability | PDF chatbot with LangSmith tracing |
| **Weeks 5-7** | Speech — STT (Whisper) + TTS | Voice assistant (speech in → speech out) |
| **Weeks 8-9** | Deployment — FastAPI + Docker + Render | Live API endpoint |
| **Weeks 10-12** | Capstone — Voice Document Assistant | Full-stack voice RAG app + blog + demo |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Orchestration** | LangChain, LangGraph (state graphs, checkpointers) |
| **LLMs** | OpenAI GPT, Hugging Face models, Llama |
| **RAG** | Chroma / Pinecone (vector DBs), LangSmith (observability) |
| **Speech** | Whisper (STT), SpeechT5 / Coqui (TTS) |
| **Deployment** | FastAPI, Streamlit, Docker, Render / Hugging Face Spaces |
| **Language** | Python 3.10+ |

---
