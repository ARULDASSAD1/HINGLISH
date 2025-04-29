from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("./hinglish-gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./hinglish-gpt2")

def generate_reply(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=50, top_p=0.65, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Sample
print(generate_reply("User: Tum kis subject mein achhe ho?\nAssistant:"))
