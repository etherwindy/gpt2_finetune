import torch
from torch.optim import AdamW
from transformers import BertTokenizer, GPT2LMHeadModel
from datasets import load_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import os
import json
import csv
import codecs
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Load the GPT2 tokenizer and model
tokenizer = BertTokenizer.from_pretrained('/storage/nvme/gpt2_finetune/gpt2-chinese-cluecorpussmall')
model = GPT2LMHeadModel.from_pretrained('/storage/nvme/gpt2_finetune/gpt2-chinese-cluecorpussmall')
tokenizer.add_tokens(['[speaker1]', '[speaker2]'])
model.resize_token_embeddings(len(tokenizer))

# Load the dataset

class MyDataset(Dataset):
    def __init__(self, tokenizer, file_path, index_file, block_size):
        self.examples = []
        self.tokenizer = tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side='right'

        block_size = block_size - self.tokenizer.num_special_tokens_to_add(pair=False)
        
        index_list = []
        with open(index_file, 'r') as f:
            for line in f:
                index_list.append(line.strip())
        
        print("reading files: ", file_path)
        dialog_list = json.loads(codecs.open(file_path, "r", "utf-8").read())
        #{'dialog_idx', 'document_id}', 'content'}
        data_list = [dialog['content'] for dialog in dialog_list if dialog['dialog_id'] in index_list]

        
        for diaglog in tqdm(data_list, desc="Reading data"):
            data = "[CLS]"
            for i, text in enumerate(diaglog):
                if i % 2 == 0:
                    data +='[speaker1]' + text + '[SEP]'
                else:
                    data += '[speaker2]' + text + '[SEP]'
            tokenized_text = tokenizer.encode(data, padding = "max_length", max_length=block_size, truncation=True, add_special_tokens=False)
            if len(tokenized_text) > block_size:
                print("Warning: data length is larger than block size")
                exit()
            self.examples.append(tokenized_text)
            
        print("number of examples: ", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
    
# Define the training function

def train(epoch, model, device, loader, optimizer, scheduler, logger):
    model.train()
    for idx, data in tqdm(enumerate(loader), total=len(loader), desc="epoch %d" % epoch):
        data = data.to(device)
        outputs = model(data, labels=data)
        loss, logits = outputs[:2]
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        logger.add_scalar("loss", loss.item(), epoch * len(loader) + idx)
            
# Set the device

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Set the hyperparameters

file_path = "data/NaturalConv/dialog_release.json"
index_file = "data/NaturalConv/train.txt"
output_dir = "output_chinese_chat"
model_dir = "/storage/nvme/gpt2_finetune/model_chinese_chat"
epoch = 1
block_size = 1024
lr = 1e-5
batch_size = 4
gradient_accumulation_steps = 2
max_grad_norm = 1.0

# Load the dataset

train_dataset = MyDataset(tokenizer, file_path, index_file, block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set the optimizer and scheduler

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=len(train_loader) * epoch)
logger = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

# Train the model

model.to(device)

for i in range(epoch):
    train(i, model, device, train_loader, optimizer, scheduler, logger)
    
# Save the model

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Save the model configuration

config = model.config
config.save_pretrained(model_dir)