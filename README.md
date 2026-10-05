# CogniAdapt AI | Deep RL & Source-Grounded Tutoring Platform

CogniAdapt AI is an advanced, multimodal educational AI companion built with **PyTorch Deep Q-Networks (DQN)**, **Bayesian Knowledge Tracing (BKT)**, **Thompson Sampling Item Calibration**, **Source-Grounded RAG with Inline Citations & Refusal Guardrails**, and **Dense Vector Semantic Deduplication**.

Developed for the **Multimodal AI Hackathon 2026 (Track D: Personalized Tutoring & Adaptive Learning)**.

## System Architecture

```
PDF
 │
 ▼
Content Ingestion Service
 │
 ▼
Chunked Content DB
 │
 ▼
LLM Quiz Generator
 │
 ▼
Quiz Questions DB
 │
 ▼
FastAPI Endpoints
 │
 ▼
Student Answers + Adaptive Difficulty
```

## Features

1. **Traceability**: Questions maintain a strict reference to the `source_chunk_id`.
2. **Clean Architecture**: Structured isolating `ingestion`, `llm`, `services`, and `api` modules.
3. **RL-Based Adaptive Difficulty Engine**: Reinforcement Learning (Q-Learning) policy matrix optimizing student engagement in the Zone of Proximal Development (ZPD target ~70-80% accuracy). State-action rewards penalize frustration & boredom while rewarding high-level mastery.
4. **Duplicate Detection Framework**: Built defensively against LLM overlapping. 
5. **Policy Analytics**: Dedicated `/rl/stats` endpoint exposing live Q-table matrix, exploration rate ($\epsilon$), and iteration tracking.

## Setup Instructions

Ensure you have Python 3.8+ installed. 

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables
Copy the environment template:
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key for `LLM_API_KEY`. If no key is provided, the system falls back to generating mock queries so you can still test the flow locally without friction!

### 3. Run the backend
Start the FastAPI server via uvicorn:
```bash
uvicorn app.main:app --reload
```

## API Testing

FastAPI provides an interactive UI out of the box. Use:

[http://localhost:8000/docs](http://localhost:8000/docs)

(FastAPI Swagger)
