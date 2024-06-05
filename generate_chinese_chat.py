import torch
import csv
import pandas as pd
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, pipeline

# Load the GPT2 tokenizer and model
tokenizer = BertTokenizer.from_pretrained('/storage/nvme/gpt2_finetune/model_chinese_chat')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model
model = GPT2LMHeadModel.from_pretrained('/storage/nvme/gpt2_finetune/model_chinese_chat')
model.eval()

# Set the device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model.to(device)

history = []

while True:
    input_sentence = input("User: ")
    if input_sentence == "exit":
        break
    history.append(input_sentence)
    
    input_text = "[CLS]"
    
    for i, text in enumerate(history):
        if i % 2 == 0:
            input_text +='[speaker1]' + text + '[SEP]'
        else:
            input_text += '[speaker2]' + text + '[SEP]'
    
    input_text += '[speaker2]'
    
    tokenized_text = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt").to(device)
    output = model.generate(tokenized_text,
                            max_length=1024,
                            pad_token_id=tokenizer.pad_token_id,
                            num_return_sequences=1,
                            temperature=0.7,
                            repetition_penalty=1.5,
                            top_k=50,
                            top_p=0.95,)
    answer = tokenizer.decode(output[0], skip_special_tokens=False)
    
    id = len(history)
    answer = answer.split("[SEP]")[id]
    answer = answer.replace("[speaker1]", "").replace("[speaker2]", "").replace(" ", "")
    print("Bot:", answer)
    history.append(answer)

print("Conversation ended.")