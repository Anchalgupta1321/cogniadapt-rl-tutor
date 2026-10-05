from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.database import engine, Base, seed_default_data
from app.api import ingest_api, quiz_api, answer_api, rl_api, bkt_api, chat_api, revision_api

# Create DB tables & seed default curriculum
Base.metadata.create_all(bind=engine)
try:
    seed_default_data()
except Exception as e:
    print(f"[Startup Seed] {e}")

app = FastAPI(title="CogniAdapt AI Engine")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Inject modularized API routes
app.include_router(ingest_api.router)
app.include_router(quiz_api.router)
app.include_router(answer_api.router)
app.include_router(rl_api.router)
app.include_router(bkt_api.router)
app.include_router(chat_api.router)
app.include_router(revision_api.router)

import os
# Create frontend directory unconditionally
os.makedirs("frontend", exist_ok=True)
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")

from fastapi.responses import RedirectResponse

@app.get("/")
def read_root():
    return RedirectResponse(url="/ui/")

