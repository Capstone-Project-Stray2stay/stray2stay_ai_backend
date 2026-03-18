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

# origins = os.getenv("BACKEND_ORIGINS", "").split(",")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins if origins != [""] else ["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.include_router(cat_router, prefix="/api/cat", tags=["Cat"])
app.include_router(dog_router, prefix="/api/dog", tags=["Dog"])

@app.get("/api/test")
def root():
    return {"message": "Hello FastAPI"}