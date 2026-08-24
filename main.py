"""Main entrypoint for running AgentOne application server."""

import uvicorn
from agentone.core.config import get_settings


def main():
    """Start uvicorn server with application settings."""
    settings = get_settings()
    print("=" * 70)
    print(f"🚀 Starting {settings.app_name}")
    print(f"📡 Serving on http://{settings.host}:{settings.port}")
    print(f"📚 OpenAPI Documentation: http://localhost:{settings.port}/docs")
    print(f"🖥️ Interactive UI Dashboard: http://localhost:{settings.port}/")
    print(f"🤖 LLM Provider: {settings.default_llm_provider}")
    print("=" * 70)

    uvicorn.run(
        "agentone.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
