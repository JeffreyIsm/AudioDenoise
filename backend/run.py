import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# python run.py
# http://localhost:8000
# http://localhost:8000/docs