import os

from dotenv import load_dotenv

load_dotenv()

WANDB_TOKEN = os.getenv("WANDB_TOKEN")
