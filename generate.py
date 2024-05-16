import torch
import csv
import pandas as pd
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load the GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('model')

# Load the model
model = GPT2LMHeadModel.from_pretrained('model')
model.eval()

# Set the device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model.to(device)

df = pd.read_csv('data/lyrics/lyrics-data.csv')
test_data = df.iloc[-500:]

input_texts = []

for index, row in test_data.iterrows():
    if row.iloc[4] != 'en':
        continue

    full_text = f"The lyrics of \"{row.iloc[1]}\":\n\n{row.iloc[3]}"
    input_text = "\n\n".join(full_text.split("\n\n")[:-1]) + "\n"
    if len(tokenizer.encode(input_text, return_tensors="pt")[0]) + 100 > 1024:
        continue
    input_texts.append(input_text)

input_text = input_texts[2]

print("Input text:")
print(input_text)

tokenized_text = tokenizer.encode(input_text, return_tensors="pt").to(device)
text_len = len(tokenized_text[0])
   
# Generate text
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
output = generator(input_text, max_length=text_len + 100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

text = output[0]['generated_text'][len(input_text):]
text = text.split("\n\n")[0]

print("Generated text:")
print(text)