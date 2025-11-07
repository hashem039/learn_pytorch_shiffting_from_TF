from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

huggingface_token = os.getenv('HF_TOKEN')

if huggingface_token:
    print("Hugging Face token successfully loaded from .env file.")
    # Example usage (uncomment and replace with your actual Hugging Face code):
    # from huggingface_hub import login
    # login(token=huggingface_token)
else:
    print("Hugging Face token not found in .env file or environment variables.")

from huggingface_hub import login
login(huggingface_token)


import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()
print('Setup is ready!')

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",  # standard model
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",  # explicitly specify tokenizer
    device=0
)
print(classifier("it was confusing"))