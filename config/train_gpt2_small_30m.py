# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

#out_dir = 'out-shakespeare-char'
eval_interval = 250 
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
#always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'owt'
wandb_run_name='small gpt'

#dataset = 'shakespeare_char'
gradient_accumulation_steps = 1

batch_size = 32
block_size = 512 # context of up to 512 previous characters

# # small GPT model
# n_layer = 12
# n_head = 12
# n_embd = 432
#dropout = 0.2

# small GPT model
n_layer = 8
n_head = 8
n_embd = 256 #
#dropout = 0.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 40000 #7500
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
beta1 = 0.9
weight_decay = 1e-1
warmup_iters = 250

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
