from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)
print(response)

#