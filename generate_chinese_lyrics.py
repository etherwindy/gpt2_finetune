import torch
import csv
import pandas as pd
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, pipeline

# Load the GPT2 tokenizer and model
tokenizer = BertTokenizer.from_pretrained('/storage/nvme/gpt2_finetune/model_chinese_lyrics')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model
model = GPT2LMHeadModel.from_pretrained('/storage/nvme/gpt2_finetune/model_chinese_lyrics')
model.eval()

# Set the device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model.to(device)

df = pd.read_csv('data/lyrics/lyrics-data.csv')
test_data = df.iloc[-500:]

file = open("data/Chinese_Lyrics/SNH48_794014/爱情养成日记_31311411.txt", "r")

input_text = file.read()
file.close()

input_text = input_text.replace("\n\n", "。").replace("\n", "。").replace(" ", "，")
input_text = "。".join(input_text.split("。")[:10]) + "。"

print("Input text:")
print(input_text)

tokenized_text = tokenizer.encode(input_text, return_tensors="pt").to(device)
text_len = len(tokenized_text[0])
   
# Generate text
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
output = generator(input_text, max_length=1024, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.5)

text = output[0]['generated_text']

print("Generated text:")
print(text)