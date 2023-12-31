from tkinter import NO
import imp
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from GptConstants import GptConstants

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from GPTLanguageModel import GPTLanguageModel
from GPTTransformers import *


class ModelHandler:

    def __init__(self) -> None:
        self.text = ""
        self.chars=""
        self.vocab_size = None
        self.train_data = None
        self.val_data = None
        self.model = None
        self.optimizer = None
        self.m = None

    def read_data(self):
        self.chars = ""
        with open('wizard_of_oz.txt','r',encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

    def encode(self, s):
        string_to_int = { ch:i for i,ch in enumerate(self.chars) }
        return [string_to_int[c] for c in s]
    
    def decode(self, l):
        int_to_string = { i:ch for i,ch in enumerate(self.chars) }
        return ''.join([int_to_string[i] for i in l])

    def train_val_data(self):        
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n=int(0.8*len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self,split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - GptConstants.block_size, (GptConstants.batch_size,))
        x = torch.stack([data[i:i+GptConstants.block_size] for i in ix])
        y = torch.stack([data[i+1:i+GptConstants.block_size+1] for i in ix])
        x = x.to(device)
        y = y.to(device)
        return x,y
    
    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(GptConstants.eval_iters)
            for k in range(GptConstants.eval_iters):
                X,Y = self.get_batch(split)
                logits, loss = self.model(X,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train_model(self):
         self.train_val_data()
         for iter in range(GptConstants.max_iters):
             print(iter)
             if iter % GptConstants.eval_iters == 0:
                 losses = self.estimate_loss()
                 print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
             # sample a batch of data
             xb, yb = self.get_batch('train')
             # evaluate the loss
             logits, loss = self.model.forward(xb, yb)
             self.optimizer.zero_grad(set_to_none=True)
             loss.backward()
             self.optimizer.step()
         self.m = self.model
         print(loss.item())

    def save_model(self):
        with open('model-01.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print('model saved')

    def start_Gpt_model(self):      
        self.model = GPTLanguageModel(self.vocab_size)
        self.model.to(device=device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=GptConstants.learning_rate)

    def load_model(self):
        print('loading model parameters...')
        with open('model-01.pkl', 'rb') as f:
            self.model = pickle.load(f)
        print('loaded successfully!')
        self.m = self.model.to(device)


    def get_output(self, prompt):
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=device)
        return self.decode(self.m.generate(context.unsqueeze(0), max_new_tokens=5)[0].tolist())
        