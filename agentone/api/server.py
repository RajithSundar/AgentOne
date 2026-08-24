"""FastAPI main application entry point and server configuration."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agentone.core.config import get_settings
from agentone.api.routes_agents import router as agents_router
from agentone.api.routes_rag import router as rag_router
from agentone.api.routes_hitl import router as hitl_router
from agentone.api.routes_eval import router as eval_router
from agentone.agents.rag_agent import get_global_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize global RAG knowledge base on startup."""
    # Warm up hybrid retriever and load knowledge documents
    get_global_retriever()
    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Enterprise-grade Autonomous Multi-Agent Operations & Intelligence Platform",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST & SSE API Routers
app.include_router(agents_router)
app.include_router(rag_router)
app.include_router(hitl_router)
app.include_router(eval_router)

# Mount Static UI directory
ui_static_dir = Path(__file__).parent.parent / "ui" / "static"
if ui_static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_static_dir)), name="static")


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health and readiness check."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "env": settings.app_env,
        "version": "1.0.0",
        "llm_provider": settings.default_llm_provider,
    }


@app.get("/", tags=["Dashboard"], include_in_schema=False)
async def serve_dashboard():
    """Serve the interactive AgentOne control center dashboard."""
    index_file = ui_static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "AgentOne API is operational. Access API docs at /docs or mount the web dashboard.",
    }
