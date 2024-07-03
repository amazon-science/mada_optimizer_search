
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


eval_interval = 2000
log_interval = 1
eval_iters = 200
hypergrad_init_iter = 0
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
out_dir = 'out_adam/'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'mada'
wandb_run_name='gpt2-124M'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 # used to simulate larger batch sizes
gradient_accumulation_steps_tmp = 1
batch_size = 12 # if gradßient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
hypergrad = True    
n_layer = 10
n_head = 10
n_embd = 480
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-3 # max learning rate
max_iters = 10000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
beta3 = 0.0
rho = 1.0
gamma = 1.0
c = 1.0
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
eps = 1e-6
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 0 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype =  'float32' 
adam = False
hyperadam = False
timestr = time.strftime("%Y%m%d-%H%M%S")
deterministic_validation = True #fixes the validation set batches
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

#directory name depends on the start time
if adam:
    out_dir = 'out_adam/' + timestr 
else:
    out_dir = 'out_mada/' + timestr
    
torch.backends.cudnn.enabled=True

seed = 5000
torch.manual_seed(seed) #originally 1337
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScalerc
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split, device, rank, cur_iter = 0, ix =None): 
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    data = train_data if split == 'train' else val_data
    del train_data, val_data
    leng = len(data)/8
    if split == 'train':
        ix = torch.randint(int(leng*(rank)), int(leng*(rank+1)) - block_size, (batch_size,))
    else:
        if deterministic_validation:
            chunk = int((len(data)-block_size)/eval_iters)
            ix = torch.arange(chunk*cur_iter, chunk*(cur_iter+1), block_size)
        else:
            ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def average_gradients_opt(mw):
    size = float(dist.get_world_size())
    for name, param in mw.optimizer.parameters.items():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def average_params_opt(mw):
    size = float(dist.get_world_size())
    for name, param in mw.optimizer.parameters.items():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def train(rank, size):

    tokens_per_iter = gradient_accumulation_steps * size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    #for some reason if I don't define it here it throws an error
    wandb_run_name='gpt2-124M'

    master_process = rank == 0
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    iter_num = 0
    best_val_loss = 1e9
    device = 'cuda:' + str(rank)
    torch.set_default_device(device)
    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value

    model.to(device)



    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split, device, rank, k)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            del X, Y
            out[split] = losses.mean()
            del losses
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        if adam:
            wandb_run_name = wandb_run_name + '-adam'
        elif hyperadam:
            wandb_run_name = wandb_run_name + '-hyperadam'
        else:
            wandb_run_name = wandb_run_name + '-mada'
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)


    # training loop
    X, Y = get_batch('train', device, rank) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model 
    running_mfu = -1.0

    #construct gdtuo wrapper
    optimizer_gdtuo = gdtuo.Meta(alpha=1e-3, beta1 = beta1, beta2 = beta2, beta3 = beta3, rho = rho, c = c, gamma = gamma, eps=1e-6,
                                optimizer=gdtuo.SGDPerParamMo(params = [['beta1', 5e-4, 0.0], ['beta2', 5e-4, 0.0], ['beta3', 1e-1, 0.0], ['rho', 1e-1, 0.0], ['c', 1e-1, 0.0], ['gamma',1e-1, 0.0], ['alpha' ,0.0 , 0.0]  ]))
 

    if init_from=='scratch':
        mw = gdtuo.ModuleWrapper(model, optimizer=optimizer_gdtuo)
    elif init_from=='resume':
        mw = checkpoint['mw']
    mw.initialize()
    #set parameters
    mw.optimizer.parameters['beta3'] = torch.tensor(beta3, device = device)
    mw.optimizer.parameters['rho'] = torch.tensor(rho, device = device)
    mw.optimizer.parameters['c'] = torch.tensor(c, device = device)
    mw.optimizer.parameters['gamma'] = torch.tensor(gamma, device = device)

    t = 0
    print(f"beta1 {Meta.clamp(mw.optimizer.parameters['beta1']).data:.4f}, beta2 {Meta.clamp(mw.optimizer.parameters['beta2'], 0.501,0.99).data:.4f}, beta3 {Meta.clamp(mw.optimizer.parameters['beta3'], 0.0, 1.0).data:.4f}, alpha {mw.optimizer.parameters['alpha'].data}")
    beta1_init = mw.optimizer.parameters['beta1']
    beta2_init = mw.optimizer.parameters['beta2']
    beta3_init = mw.optimizer.parameters['beta3']
    rho_init = mw.optimizer.parameters['rho']
    c_init = mw.optimizer.parameters['c']
    gamma_init = mw.optimizer.parameters['gamma']
    lr = get_lr(iter_num)
    while True:

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if wandb_log and rank == 0:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "beta1": mw.optimizer.parameters['beta1'],
                    "beta2": mw.optimizer.parameters['beta2'],
                    "beta3": mw.optimizer.parameters['beta3'],
                    "rho": mw.optimizer.parameters['rho'],
                    "c": mw.optimizer.parameters['c'],
                    "gamma": mw.optimizer.parameters['gamma']
                })
            if (losses['val'] < best_val_loss or always_save_checkpoint) and rank == 0:
                print(mw.module.get_num_params())
                print(model.get_num_params())
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_mw': mw.optimizer,
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'training_loss': losses['train'],
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt' + str(iter_num) + '.pt'))
                    

        if iter_num == 0 and eval_only:
            break

        mw.begin()
        if iter_num == 0: #initialize the grads in the first iteration
            mw.zero_grad()

        logits, loss = mw.forward(X, Y)
        loss = loss/gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train', device, rank)
        loss.backward()
                    
        # we are not updating the lr
        mw.optimizer.parameters['alpha'].grad = torch.zeros_like(mw.optimizer.parameters['alpha'])
        if iter_num < hypergrad_init_iter or adam:
            for n, p in mw.optimizer.parameters.items():
                p.grad = torch.zeros_like(p)

        elif hyperadam:
            
            for n, p in mw.optimizer.parameters.items():
                if n != 'beta1' and n != 'beta2':
                    p.grad = torch.zeros_like(p)
        
        alpha_temp = mw.optimizer.parameters['alpha'].data
        
        #if previous iteration was a model update iteration, update optimizer state
        if not adam and (((iter_num-1) % gradient_accumulation_steps == 1) or gradient_accumulation_steps == 1): #mw.optimizer.parameters['beta1'].grad is not None and mw.optimizer.parameters['beta1'].grad.data != torch.tensor(0.0) and not adam:

            average_gradients_opt(mw)
            for n,v in mw.optimizer.parameters.items():
                torch.nn.utils.clip_grad_norm_(v,10.0)
            alpha_temp = mw.optimizer.parameters['alpha'].data
            mw.optimizer.parameters['alpha'].data = torch.tensor(0.0) #to make sure model is not updated
            mw.optimizer.optimizer.step(mw.optimizer.parameters)

            mw.optimizer.parameters['alpha'].data = alpha_temp
            mw.optimizer.zero_grad()

        
        if iter_num % gradient_accumulation_steps != 1 and gradient_accumulation_steps != 1:
            mw.optimizer.parameters['alpha'].data = alpha_temp
            mw.detach()
        #if current iteration is an update iteration
        if iter_num % gradient_accumulation_steps == 1 or gradient_accumulation_steps == 1:
            #get the correct learning rate
            lr = get_lr(iter_num) if decay_lr else learning_rate
            mw.optimizer.parameters['alpha'].data = torch.tensor(lr)

            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            #manual weight decay:
            
            for p in model.parameters():
                if p.data.dim() >= 2 and p.requires_grad:
                    p.data.copy_(p.data - mw.optimizer.parameters['alpha']*weight_decay*p.data)
            
            #gradient averaging
            average_gradients(model)
            mw.step()
            mw.zero_grad()                
        
        device_id =  device[-1]
        # timing and logging
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size*gradient_accumulation_steps*size, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            if rank == 0:
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%") 
                print(f"beta1 {Meta.clamp(mw.optimizer.parameters['beta1']).data:.4f}, beta2 {Meta.clamp(mw.optimizer.parameters['beta2'], 0.51,0.99).data:.4f}, beta3 {Meta.clamp(mw.optimizer.parameters['beta3'],0.0,1.0).data:.4f}, rho {Meta.clamp(mw.optimizer.parameters['rho'],0.0,1.0).data:.4f}, c {Meta.clamp(mw.optimizer.parameters['c'],0.0,1.0).data:.4f}, gamma {Meta.clamp(mw.optimizer.parameters['gamma'],0.0,1.0).data:.4f}, alpha {mw.optimizer.parameters['alpha'].data}")#, hyper alpha {mw.optimizer.optimizer.parameters['alpha']}")
                res = np.array([mw.optimizer.parameters['beta1'].cpu().detach(), mw.optimizer.parameters['beta2'].cpu().detach(), mw.optimizer.parameters['beta3'].cpu().detach(), mw.optimizer.parameters['rho'].cpu().detach(), mw.optimizer.parameters['c'].cpu().detach(), mw.optimizer.parameters['gamma'].cpu().detach(), losses['train'].cpu(), losses['val'].cpu()])
                res = np.reshape(res,(1,8))
                if hypergrad and rank == 0:
                    if adam:
                        with open( out_dir + '/traj_owt_gpt2_adam' + '.txt', 'a+') as f:
                            np.savetxt(f, res ,delimiter =', ', fmt='%f')
                    else:
                        with open( out_dir + '/traj_owt_gpt2_mada' + '.txt', 'a+') as f:
                            np.savetxt(f, res ,delimiter =', ', fmt='%f')
        iter_num += 1
        local_iter_num += 1

        
        # termination conditions
        if iter_num > max_iters or math.isnan(losses['train']):
            losses = estimate_loss()
            
            print(device_id)
            print(f"iter {iter_num}: loss {losses['val'].item():.4f}") 
            print(f"beta1 {mw.optimizer.parameters['beta1']:.4f}, beta2 {mw.optimizer.parameters['beta2']:.4f}, beta3 {mw.optimizer.parameters['beta3']:.4f}, alpha {mw.optimizer.parameters['alpha']}")
            res = np.array([beta1_init.detach().cpu(), beta2_init.detach().cpu(), beta3_init.detach().cpu(), rho_init.detach().cpu(), c_init.detach().cpu(), gamma_init.detach().cpu(), losses['train'].cpu(), losses['val'].cpu()])
            res = np.reshape(res,(1,8))
            if hypergrad and rank == 0:
                if adam:
                    with open(out_dir + '/hyper_train_log_owt_gpt2_adam.txt', 'a+') as f:
                        np.savetxt(f, res ,delimiter =', ', fmt='%f')
                else:
                    with open(out_dir + '/hyper_train_log_owt_gpt2_mada.txt', 'a+') as f:
                        np.savetxt(f, res ,delimiter =', ', fmt='%f')

            if losses['val'] < best_val_loss and rank == 0:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'mw': mw,
                        'optimizer': mw.optimizer,
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'training_loss': losses['train'],
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt')) #('ckpt_{}.pt'.format(int(time.time())))


            break


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.0'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 8
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

