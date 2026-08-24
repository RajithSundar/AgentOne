# Autonomous Build Protocol & Engineering Standards

## 1. Multi-Agent & RAG Stack
- Framework: LangGraph (StateGraph, compiled with memory checkpointer), LangChain Core.
- API Layer: FastAPI (async REST endpoints + Server-Sent Events / SSE for live agent streaming).
- LLM Providers: Google Gemini (`google-genai`), OpenAI, Anthropic, and deterministic Mock Provider for zero-dependency testability.
- Knowledge Subsystem: Hybrid Search (Dense Cosine Similarity + Sparse BM25 + Reciprocal Rank Fusion).
- Safety & Governance: Input Shield (PII Redaction, Prompt Injection Defense), Output Verifier (Pydantic Schema Validation, Groundedness Check).
- Evaluation & Telemetry: Faithfulness, Answer Relevancy, Context Precision/Recall, OpenTelemetry-compatible tracing.

## 2. Code Quality & Git Decorum
- Strictly typed Python 3.11+ using Pydantic v2 and Python typing.
- Modular decoupled architecture (Core, Agents, RAG, Guardrails, HITL, Eval, API, UI).
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `ci:`).
- Human-written, professional software engineering style with clean docstrings and zero generic AI boilerplate.
