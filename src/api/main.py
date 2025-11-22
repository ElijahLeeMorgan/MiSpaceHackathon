"""FastAPI application and configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from dotenv import load_dotenv
from pathlib import Path

from src.api import routes

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="MISpace Hackathon API",
    description="Data Science Project API with analytics and visualization endpoints",
    version="0.1.0",
)

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_path = Path(__file__).parent.parent / "frontend"
if (frontend_path / "static").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

# Include routers
app.include_router(routes.router)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard page."""
    index_path = frontend_path / "templates" / "index.html"
    if index_path.exists():
        with open(index_path, "r") as f:
            return f.read()
    return """
    <html>
        <head><title>MISpace Hackathon</title></head>
        <body>
            <h1>Welcome to MISpace Hackathon API</h1>
            <p>Visit <a href="/docs">/docs</a> for API documentation</p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "True").lower() == "true"
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )
