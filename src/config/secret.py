import os

from dotenv import load_dotenv

load_dotenv()

WANDB_TOKEN = os.getenv("WANDB_TOKEN")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")
GSERVICE_KEY_PATH = os.getenv("GSERVICE_KEY_PATH")
GOAUTH_KEY_PATH = os.getenv("GOAUTH_KEY_PATH")