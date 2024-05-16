import torch
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import os
import json
import csv
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Load the GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the dataset

class MyDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.examples = []
        self.tokenizer = tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side='right'

        block_size = block_size - self.tokenizer.num_special_tokens_to_add(pair=False)
        print("reading file: ", file_path)
                
        df = pd.read_csv(file_path)
                
        train_data = df.iloc[:-500]
                
        for index, row in train_data.iterrows():
            if row.iloc[4] != 'en':
                continue
            text = f"The lyrics of \"{row.iloc[1]}\":\n\n{row.iloc[3]}"
            tokenized_text = self.tokenizer.encode(text, padding="max_length", max_length=block_size, truncation=True)
            self.examples.append(tokenized_text)
        print("number of examples: ", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
    
# Define the training function

def train(epoch, model, device, loader, optimizer, scheduler, logger):
    model.train()
    for idx, data in tqdm(enumerate(loader), total=len(loader)):
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

file_path = "data/lyrics/lyrics-data.csv"
output_dir = "output"
model_dir = "model"
epoch = 1
block_size = 1024
lr = 3e-5
batch_size = 4
gradient_accumulation_steps = 4
max_grad_norm = 1.0

# Load the dataset

train_dataset = MyDataset(tokenizer, file_path, block_size)
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