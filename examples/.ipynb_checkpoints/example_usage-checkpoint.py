"""
Example showing how to use the Chain of Thought (CoT) Decoding implementation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.decoding import CoT_Decoding

def main():
    # Model and tokenizer initialization
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" # Replace with your preferred model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model and tokenizer from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Decoding parameters
    k = 5  # Number of top tokens to consider
    max_new_tokens = 512
    
    print("Running Chain of Thought decoding...")
    answer, confidence = CoT_Decoding(
        messages=messages,
        k=k,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        agg_path=False
    )
    
    print(f"\nAnswer: {answer}")
    print(f"Confidence Score: {confidence:.4f}")

if __name__ == "__main__":
    main()