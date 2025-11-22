from mlx_lm import load, generate

# 1. Load a lightweight model optimized for Mac
# Llama-3.2-1B is incredibly fast and small (runs on any M-chip)
model_id = "mlx-community/Llama-3.2-3B-Instruct-4bit"
model, tokenizer = load(model_id)

def get_sentiment(text):
    # 2. Construct a strict prompt
    prompt = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Classify the sentiment of the following text as 'Positive', 'Negative', or 'Neutral'. 
    Reply ONLY with the one word label.
    
    Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    # 3. Generate the label
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=5,  # We only need one word
        verbose=False
    )
    return response.strip()

# Test it
reviews = [
    "La calidad de la máquina es increíble y me encanta!",
    "La máquina funciona durante tres selecciones y se daña.",
    "Llegaste a tiempo pero a tiempo es tarde, 5 minutos antes es a tiempo, y tarde es perdidas!!."
]

print("--- Sentiment Analysis Results ---")
for review in reviews:
    sentiment = get_sentiment(review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")