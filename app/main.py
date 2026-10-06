import os
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

from routers.cat_router import router as cat_router
from routers.dog_router import router as dog_router

import pillow_heif

pillow_heif.register_heif_opener()

load_dotenv()

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    )

origins = os.getenv("BACKEND_ORIGINS", "").split(",")
origins = [o.strip() for o in origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Origin", "Content-Type", "Authorization"],
)

app.include_router(cat_router, prefix="/api/cat", tags=["Cat"])
app.include_router(dog_router, prefix="/api/dog", tags=["Dog"])

@app.get("/api/test")
def root():
    return {"message": "Hello FastAPI"}