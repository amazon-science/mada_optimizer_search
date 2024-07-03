from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm
import sys
import os
import time
import math
import pickle
from contextlib import nullcontext

from matplotlib.lines import Line2D  
import matplotlib.pyplot as plt

import torch.multiprocessing as mp

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

cwd = os.getcwd()
import gdtuo
from gdtuo import Meta



from model import GPTConfig, GPT

# model
hypergrad = True    
n_layer = 12
n_head = 12
n_embd = 768
batch_size = 30
block_size = 1024
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
device = "cuda:0"
model_id = "gpt2"
model_hf = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
torch.manual_seed(5000) 
out_dir = "/fsx/nanoGPT/out_mada/20230922-204824/"
ckpt_path = out_dir + "ckpt500000.pt"
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
model_args = dict()
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]

# # create the model
model_args['vocab_size'] = 50304 # always 50257 for GPT model checkpoints

config = GPTConfig(**model_args)
model = GPT(config)
model.to(device)
sd = checkpoint['model_state_dict']
model.load_state_dict(sd, strict = True)


datasets = ["wikitext", "openwebtext", "lambada"] 
device_type = "cuda"
eval_iters = 200
def get_batch(split, device, rank, dataset, eval_iters, cur_iter, ix =None): 
    data_dir = os.path.join('/fsx/nanoGPT/data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    data = train_data if split == 'train' else val_data
    del train_data, val_data
    leng = len(data)/8
    if split == 'train':
        ix = torch.randint(int(leng*(rank)), int(leng*(rank+1)) - block_size, (batch_size,))
    else:
        chunk = int((len(data)-block_size)/eval_iters)
        #ix = torch.randint(len(data) - block_size, (batch_size,))
        ix = torch.arange(chunk*cur_iter, chunk*(cur_iter+1), block_size)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(dataset):
    out = {}
    model.eval()
    cur_iter = 0
    for split in ['val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, device, 0, dataset, eval_iters, cur_iter)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            cur_iter += 1
        del X, Y
        out[split] = losses.mean()
        del losses
    model.train()
    return out

#make the measurements
for d in datasets:
    losses = estimate_loss(d)
    print(d)
    print(losses)
    ppl = torch.exp(losses['val'])
    print(ppl)
