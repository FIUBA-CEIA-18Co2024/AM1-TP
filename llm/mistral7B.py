import os
import torch
from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Set device to CUDA if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_hotel_reviews(prompt, num_reviews=5):

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device=device,
        token=HF_TOKEN
    )

    reviews = []
    for _ in range(num_reviews):
        prompt_template = f"""
        Generate a realistic hotel review with the following characteristics:
        {prompt}
        Make it sound natural and include specific details.
        Review:
        """

        response = pipe(
            prompt_template,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.85,
            num_return_sequences=1
        )

        if response and len(response) > 0:
            generated_text = response[0].get('generated_text', '')
            reviews.append(generated_text)

    return reviews

# Example usage
prompts = [
    "Rating: 2/5, Issues: cleanliness and service",
    "Rating: 5/5, Highlights: location and breakfast",
    "Rating: 3/5, Mixed experience: good location but noisy"
]

for prompt in prompts:
    reviews = generate_hotel_reviews(prompt)
    print(f"\nPrompt: {prompt}")
    for i, review in enumerate(reviews, 1):
        print(f"\nReview {i}:", review)
