import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'best_model.h5')

print("MODEL_PATH:", MODEL_PATH)

model = load_model(MODEL_PATH)