import os

# Define the folder where files will be saved
STATIC_DIR_NAME = "static"
STATIC_DIR_PATH = os.path.join(os.path.dirname(__file__), STATIC_DIR_NAME)
STATIC_URL_PATH = "/static"