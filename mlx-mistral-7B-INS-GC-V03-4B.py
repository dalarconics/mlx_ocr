import mlx.core as mx
from mlx_lm import load, generate
import gc

def run_model(model_id, prompt):
    print(f"\n{'='*40}")
    print(f"⬇️  Loading: {model_id}")
    print(f"{'='*40}")
    
    # 1. Load Model & Tokenizer
    model, tokenizer = load(model_id)
    
    # 2. Generate
    print(f"🤖 Generating response for: '{prompt}'")
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=200, 
        verbose=True 
    )
    print(f"\n✅ Output:\n{response}")

    # 3. CLEANUP (Crucial for 18GB RAM)
    # We delete the model from memory so we have space for the next one
    del model
    del tokenizer
    gc.collect() # Force Python to release RAM
    print("🧹 Memory cleaned.")

def main():
    # --- MODEL 1: MISTRAL (The "Little Brother") ---
    # Mistral 7B v0.3 is excellent at following instructions.
    # 4-bit size: ~4.5 GB
    mistral_id = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    run_model(mistral_id, "Write a one-sentence joke about Python programming.")

if __name__ == "__main__":
    main()