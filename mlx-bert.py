import mlx.core as mx
from mlx_embeddings import load

# 1. Load the model and tokenizer
# This returns the raw MLX neural network and the tokenizer
model, tokenizer = load("mlx-community/all-MiniLM-L6-v2-4bit")

def generate_embedding(text):
    # 2. Tokenize the text
    # We add special tokens (like [CLS] and [SEP]) which BERT needs
    tokens = tokenizer.encode(text)
    
    # 3. Convert to MLX Array (Shape: [1, length])
    input_ids = mx.array([tokens])
    
    # 4. Run the model
    # This returns a hidden state for every token in your sentence
    outputs = model(input_ids)
    
    # The output is usually the 'last_hidden_state'
    # Shape: [batch_size, seq_length, 384]
    last_hidden_state = outputs.last_hidden_state
    
    # 5. Pooling (The "Average")
    # To get 1 vector for the whole sentence, we average all token vectors.
    # axis=1 means "average across the sentence length"
    embedding = mx.mean(last_hidden_state, axis=1)
    
    return embedding

# Test it
text = "This is a test sentence."
vector = generate_embedding(text)

print(f"Success! Generated vector of shape: {vector.shape}")
print(f"First 10 values: {vector[0, :10]}")