from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload
from fastapi.staticfiles import StaticFiles
from app.utils import STATIC_DIR_PATH, STATIC_URL_PATH

app = FastAPI()

app.mount(STATIC_URL_PATH, StaticFiles(directory=STATIC_DIR_PATH), name='static')

origins = [
    #url of frontend servers, 
    #any url that can access the backend
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://audio-denoise.vercel.app",
]

#Cors prohibits unauthorized from accessing API
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.include_router(
    upload.router,
)