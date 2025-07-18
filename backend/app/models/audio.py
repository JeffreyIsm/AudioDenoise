'''
defines what fastapi endpoint should return,
in a validated format.
Not strictly required, but can make API more explicit
and documented in /docs

called in @router in upload.py
'''
from pydantic import BaseModel

class AudioResponse(BaseModel):
    original_url: str
    denoised_url: str

'''
For dynamic files (like denoised audio):
Use a cloud storage service (S3, GCS, etc.) to store and serve files.

which ones free
'''