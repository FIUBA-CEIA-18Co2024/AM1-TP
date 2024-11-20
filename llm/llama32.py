import os
import torch
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
model_id = "meta-llama/Llama-3.2-1B"

# Set device to CUDA if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device=device,  # Explicitly set device
    token=HF_TOKEN
)

pipe("The key to life is")
