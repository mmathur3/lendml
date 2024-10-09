from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Encode input text
input_text = "Hello, world!"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate output
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated text:")
print(output_text)
