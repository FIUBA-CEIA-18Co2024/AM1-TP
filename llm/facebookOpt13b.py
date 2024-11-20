import os
import torch
from dotenv import load_dotenv
from transformers import pipeline
import gc

def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        print(f"Current GPU memory usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

free_memory()

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
model_id = "facebook/opt-1.3b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def generate_hotel_reviews(prompt, num_reviews=5):
    # Initialize pipeline outside the loop
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.float16,
        device=device,
        token=HF_TOKEN
    )

    reviews = []
    for i in range(num_reviews):
        # Improved prompt template
        prompt_template = f"""Write a detailed hotel review.
Rating and Details: {prompt}
Please provide specific examples and details in your review.

Review:"""

        try:
            # Generate response with more explicit parameters
            response = pipe(
                prompt_template,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.85,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=pipe.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

            if response:
                # Extract and clean the generated text
                generated_text = response[0]['generated_text']
                # Remove the prompt from the response
                review_text = generated_text.split("Review:")[1].strip() if "Review:" in generated_text else generated_text
                reviews.append(review_text)

                print(f"Generated review {i+1}: {review_text[:100]}...")  # Print first 100 chars for debugging

        except Exception as e:
            print(f"Error generating review: {str(e)}")
            continue

        # Free memory after each generation
        free_memory()

    return reviews

# Test prompts
prompts = [
    "Rating: 2/5, Issues: cleanliness and service",
    "Rating: 5/5, Highlights: location and breakfast",
    "Rating: 3/5, Mixed experience: good location but noisy"
]

# Generate and print reviews
for prompt in prompts:
    print(f"\n\nGenerating reviews for prompt: {prompt}")
    reviews = generate_hotel_reviews(prompt, num_reviews=2)  # Reduced to 2 reviews for testing

    print(f"\nPrompt: {prompt}")
    for i, review in enumerate(reviews, 1):
        print(f"\nReview {i}:")
        print("-" * 50)
        print(review)
        print("-" * 50)
