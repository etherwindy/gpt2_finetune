from bert_score import score
import torch
import csv
import os
import random
import pandas as pd
from torch.nn import functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('/storage/nvme/gpt2_finetune/gpt2')

original_model = GPT2LMHeadModel.from_pretrained('/storage/nvme/gpt2_finetune/gpt2').to(device)
fintuned_model = GPT2LMHeadModel.from_pretrained('/storage/nvme/gpt2_finetune/model').to(device)

file_path = 'data/Chinese_Lyrics'

test_data = []

for singer in tqdm(os.listdir(file_path)[-1:], desc="Reading data"):
    if not os.path.isdir(os.path.join(file_path, singer)):
        continue
    for song in os.listdir(os.path.join(file_path, singer)):
        with open(os.path.join(file_path, singer, song), 'r') as f:
            data = f.read()
            data = data.replace("\n\n", "。").replace("\n", "。").replace(" ", "，")
            test_data.append(data)

reference_texts = []
original_output_texts = []
fintuned_output_texts = []

original_model.eval()
fintuned_model.eval()

original_generator = pipeline('text-generation', model=original_model, tokenizer=tokenizer, device=device)
fintuned_generator = pipeline('text-generation', model=fintuned_model, tokenizer=tokenizer, device=device)

for index, text in tqdm(enumerate(test_data), total=len(test_data), desc="Generating text"):
    
    splited_text = text.split("。")

    #生成随机数，范围为(1, len(splited_text)-1)
    
    n = random.randint(1, len(splited_text)-1)    
    
    input_text = "。".join(splited_text[:n]) + "。"
    reference_text = "。".join(splited_text[n:])

    input_len = len(tokenizer.tokenize(input_text))
    if input_len + 100> 1024:
        continue
    
    tokenized_text = tokenizer.encode(input_text, return_tensors="pt")
    
    tokenized_text = tokenized_text.to(device)

    original_output = original_generator(input_text, max_length=1024, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    fintuned_output = fintuned_generator(input_text, max_length=1024, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    original_output_text = original_output[0]['generated_text'][len(input_text):]
    fintuned_output_text = fintuned_output[0]['generated_text'][len(input_text):]
    
    #if original_output_text.strip() == "" or fintuned_output_text.strip() == "" or reference_text.strip() == "":
    #    print(index, "empty output")
    #    continue
    
    reference_texts.append(reference_text)
    original_output_texts.append(original_output_text)
    fintuned_output_texts.append(fintuned_output_text)

print("Number of valid test examples: ", len(reference_texts))
    
original_P, original_R, original_F1 = score(original_output_texts, reference_texts, lang='cn', verbose=True, model_type='bert-base-chinese', device=device)
finetuned_P, finetuned_R, finetuned_F1 = score(fintuned_output_texts, reference_texts, lang='cn', verbose=True, model_type='bert-base-chinese', device=device)

print("===========================================")
print("Original model:")
print("Precision: ", original_P.mean())
print("Recall: ", original_R.mean())
print("F1: ", original_F1.mean())
print("===========================================")
print("Finetuned model:")
print("Precision: ", finetuned_P.mean())
print("Recall: ", finetuned_R.mean())
print("F1: ", finetuned_F1.mean())
print("===========================================")


    
