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
    # --- MODEL 2: LLAMA 3.1 8B (The Standard) ---
    # Llama 3.1 8B is currently the state-of-the-art for this size.
    # 4-bit size: ~5.5 GB
    llama_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    run_model(llama_id, "Explain why the sky is blue in simple terms.")

if __name__ == "__main__":
    main()