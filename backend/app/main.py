# from fastapi import FastAPI
# from app.api.routes import router

# app = FastAPI(title="Speech Emotion Recognition API")

# app.include_router(router)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(title="Speech Emotion Recognition API")

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)