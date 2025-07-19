import os
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.h5')

model = load_model(MODEL_PATH)