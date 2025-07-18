'''
This is the API calls, directing the files to go placess
'''

from fastapi import APIRouter, UploadFile, File
from app.services.audio_denoise import process_audio
import os
import uuid
from app.models.audio import AudioResponse
from app.utils import STATIC_DIR_PATH, STATIC_URL_PATH

router = APIRouter()

@router.post("/upload", response_model=AudioResponse)
async def upload(file: UploadFile = File(...)): #expects file from POST
    #creates safe and unique filename
    file_ext = file.filename.split('.')[-1]
    unique_fn = f"{uuid.uuid4()}.{file_ext}"
    raw_path = os.path.join(STATIC_DIR_PATH, unique_fn)

    #writes uploaded file to disk as binary wb
    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    #pass saved file to denoising logic in /services
    result_path = process_audio(raw_path)

    #React gets this URL and can preview/download the denoised audio
    return {
        "original_url": f"{STATIC_URL_PATH}/{os.path.basename(raw_path)}",
        "denoised_url": f"{STATIC_URL_PATH}/{os.path.basename(result_path)}",
        }