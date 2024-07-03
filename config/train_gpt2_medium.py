# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'mada'
wandb_run_name='gpt2-355M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# 6*1024*1*8 = 49,804 
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 #was 5
bias = False

n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0

# this makes total number of tokens be 300B
learning_rate = 5e-4 # with baby networks can afford to go a bit higher
max_iters = 100000*gradient_accumulation_steps #7500
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1.5e-5 
beta2 = 0.95 
beta1 = 0.9
weight_decay = 1e-1
warmup_iters = 2000*gradient_accumulation_steps

# eval stuff
eval_interval = 500*gradient_accumulation_steps
eval_iters = 200
log_interval = 10*gradient_accumulation_steps
hypergrad_init_iter = 50*gradient_accumulation_steps


