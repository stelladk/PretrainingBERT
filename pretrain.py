from tokenizer import createTokenizer
from preprocessing import process_text
from transformers import BertConfig, BertForMaskedLM

from transformers import RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch


import time
import argparse
# import glob
# import csv
# import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
# tqdm.pandas()


parse = argparse.ArgumentParser()
parse.add_argument('--tokenizer', type=str, help='tokenizer path if you want to use an existing one')
parse.add_argument('--files', type=str, help='files of raw text')
parse.add_argument('--encoding', type=str, default="utf8", help='text encoding')
parse.add_argument('--vocab_size', type=int, default=32000, help='vocabulary size for tokenizer')
parse.add_argument('--min_freq', type=int, default=2, help='minimum term frequency for the vocabulary')
parse.add_argument('--sequence_len', type=int, default=512, help='maximum length of embeddings')
parse.add_argument('--model_path', type=str, default='', help='path to save the pretrained model')
parse.add_argument('--dataset', type=str, default=None, help='path to existing .pt dataset file')
parse.add_argument('--dataset_name', type=str, default='new_dataset', help='dataset name if it does not exist')
parse.add_argument('--mlm_prob', type=float, default=0.15, help='mlm masking probability')

parse.add_argument('--hidden_layers', type=int, default=12, help='number of hidden layers of BERT model')
parse.add_argument('--hidden_size', type=int, default=768, help='hidden size of BERT model')
parse.add_argument('--attention_heads', type=int, default=12, help='number of attention heads of BERT model')

parse.add_argument('--epochs', type=int, default=40, help='number of epochs')
parse.add_argument('--batch_size', type=int, default=8, help='training batch size')
parse.add_argument('--max_steps', type=int, default=0, help='number of training steps, overwrites epochs')
parse.add_argument('--lrate', type=float, default=1e-4, help='learning rate')
parse.add_argument('--b1', type=float, default=0.9, help='adam beta1 parameter')
parse.add_argument('--b2', type=float, default=0.99, help='adam beta2 parameter')
parse.add_argument('--wdecay', type=float, default=0.01, help='weight decay')
parse.add_argument('--scheduler', type=str, default='linear', help='learning rate scheduler type')
parse.add_argument('--warmup_steps', type=int, default=10_000, help='warmup steps')

parse.add_argument('--checkpoint', type=str, help='path to save the checkpoints')
parse.add_argument('--save_steps', type=int, default=10_000, help='save checkpoint every # of steps')
parse.add_argument('--save_limit', type=int, default=5, help='limit of checkpoints')

parse.add_argument('--resume', type=str, default=None, help='resume from checkpoint path')

args = parse.parse_args()

print()

# Create directories
try:
    os.mkdir(args.model_path)
except OSError as err: 
    print()

# Create Tokenizer
tokenizer_path = None
if args.tokenizer==None :
    print("Creating new tokenizer")
    tokenizer_path = args.model_path+"/tokenizer/"
    try:
        os.mkdir(tokenizer_path)
    except OSError as err:
        print()
    createTokenizer(args.files, args.vocab_size, args.min_freq, args.sequence_len, tokenizer_path)
else:
    tokenizer_path = args.tokenizer
    print("Using tokenizer from", tokenizer_path)



# Load Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, max_len=args.sequence_len)

# Create lamda tokenizing function
def map_tokenize(text):
    return tokenizer.encode(text, max_length=args.sequence_len, truncation=True)


# Process Text
dataset_path = None
if args.dataset == None :
    print("Processing text")
    dataset_path = process_text(args.files, args.dataset_name, map_tokenize, args.encoding)
else:
    dataset_path = args.dataset
    print("Using dataset from", dataset_path)

# Load Dataset
dataset = torch.load(dataset_path) 



# Create Masked Language Model
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
)
data_collator


config = BertConfig(
    vocab_size=args.vocab_size,
    max_position_embeddings=args.sequence_len,
    num_hidden_layers=args.hidden_layers,    #L
    hidden_size=args.hidden_size,        #H
    num_attention_heads=args.attention_heads,  #A
    type_vocab_size=1,
)


model = BertForMaskedLM(config=config)


training_args = TrainingArguments(
    output_dir=args.checkpoint,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    save_steps=args.save_steps,
    save_total_limit=args.save_limit,
    prediction_loss_only=True,
    max_steps=args.max_steps,
    learning_rate=args.lrate,
    adam_beta1=args.b1,
    adam_beta2=args.b2,
    weight_decay=args.wdecay,
    lr_scheduler_type=args.scheduler,
    warmup_steps=args.warmup_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train
if args.resume == None :
    print("Pre-training BERT model")
    trainer.train()
else:
    print("Pre-training BERT model from checkpoint", args.resume)
    trainer.train(resume_from_checkpoint=args.resume)

# Save model
print("Saving model at", args.model_path)
trainer.save_model(args.model_path)

#TODO: Pre-train from model?