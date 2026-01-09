from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Steam Oracle API",
        description="Agentic search engine backend",
        version="1.0.0"
    )
    
    # Configure CORS - Allow everything for development ease
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # In production, verify specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include Routers
    app.include_router(api_router, prefix="/api/v1")
    
    @app.get("/health")
    def health_check():
        return {"status": "ok", "version": "1.0.0"}
        
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Hot reload enabled for dev
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
