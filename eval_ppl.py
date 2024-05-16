import torch
from torch.nn import CrossEntropyLoss
import math
import csv
import pandas as pd
from torch.nn import functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 计算困惑度

def calculate_ppl(model_name, test_data, device):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    model.eval()
    total_loss = 0.0
    total_length = 0

    with torch.no_grad():
        for text in tqdm(test_data):
            # Tokenize input text and check length
            tokenized_text = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
            text_len = tokenized_text.size(1)

            # Forward pass
            outputs = model(tokenized_text, labels=tokenized_text)
            loss = outputs.loss

            # Accumulate loss and length
            total_loss += loss.item() * text_len
            total_length += text_len

    # Calculate perplexity
    avg_loss = total_loss / total_length
    ppl = math.exp(avg_loss)
    return ppl

# 测试数据

df = pd.read_csv('data/lyrics/lyrics-data.csv')
test_rows = df.iloc[-500:]

test_data = []
for index, row in test_rows.iterrows():
    if row.iloc[4] == 'en':
        text = f"The lyrics of \"{row.iloc[1]}\":\n{row.iloc[3]}"
        test_data.append(text)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 计算原模型的PPL
original_model_ppl = calculate_ppl('gpt2', test_data, device)
print(f'Original model PPL: {original_model_ppl}')

# 计算微调后模型的PPL
fine_tuned_model_ppl = calculate_ppl('model', test_data, device)
print(f'Fine-tuned model PPL: {fine_tuned_model_ppl}')