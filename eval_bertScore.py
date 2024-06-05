from bert_score import score
import torch
import csv
import pandas as pd
from torch.nn import functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('/storage/nvme/gpt2_finetune/gpt2')

original_model = GPT2LMHeadModel.from_pretrained('/storage/nvme/gpt2_finetune/gpt2').to(device)
fintuned_model = GPT2LMHeadModel.from_pretrained('/storage/nvme/gpt2_finetune/model').to(device)

df = pd.read_csv('data/lyrics/lyrics-data.csv')
test_data = df.iloc[-500:]

reference_texts = []
original_output_texts = []
fintuned_output_texts = []

original_model.eval()
fintuned_model.eval()

original_generator = pipeline('text-generation', model=original_model, tokenizer=tokenizer, device=device)
fintuned_generator = pipeline('text-generation', model=fintuned_model, tokenizer=tokenizer, device=device)

for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    
    if row.iloc[4] != 'en':
        continue
    
    full_text = f"The lyrics of \"{row.iloc[1]}\":\n\n{row.iloc[3]}"
    splited_text = full_text.split("\n\n")
    input_text = "\n\n".join(splited_text[:-1]) + "\n"
    reference_text = splited_text[-1]

    input_len = len(tokenizer.tokenize(input_text))
    if input_len + 100> 1024:
        continue
    
    tokenized_text = tokenizer.encode(input_text, return_tensors="pt")
    text_len = len(tokenized_text[0])
    
    tokenized_text = tokenized_text.to(device)

    original_output = original_generator(input_text, max_length=text_len + 100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    fintuned_output = fintuned_generator(input_text, max_length=text_len + 100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    original_output_text = original_output[0]['generated_text'][len(input_text):]
    fintuned_output_text = fintuned_output[0]['generated_text'][len(input_text):]
    
    original_output_text = original_output_text.split("\n\n")[0]
    fintuned_output_text = fintuned_output_text.split("\n\n")[0]
    
    #if original_output_text.strip() == "" or fintuned_output_text.strip() == "" or reference_text.strip() == "":
    #    print(index, "empty output")
    #    continue
    
    reference_texts.append(reference_text)
    original_output_texts.append(original_output_text)
    fintuned_output_texts.append(fintuned_output_text)

print("Number of valid test examples: ", len(reference_texts))
    
original_P, original_R, original_F1 = score(original_output_texts, reference_texts, lang='en', verbose=True, model_type='bert-base-uncased', device=device)
finetuned_P, finetuned_R, finetuned_F1 = score(fintuned_output_texts, reference_texts, lang='en', verbose=True, model_type='bert-base-uncased', device=device)

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


    
