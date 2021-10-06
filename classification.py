from transformers import BertModel, CamembertModel, CamembertTokenizer, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
import torch
# import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

import time
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm


parse = argparse.ArgumentParser()
parse.add_argument('--model', type=str, default='camembert', help='camembert, barthez or custom')
parse.add_argument('--pretrained_path', type=str, default='camembert-base', help='pretrained BERT model path')
parse.add_argument('--tokenizer_path', type=str, default='camembert-base', help='pretrained tokenizer path')

parse.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parse.add_argument('--sequence_len', type=int, default=512, help='maximum sequence length')
parse.add_argument('--batch_size', type=int, default=8, help='batch size')
parse.add_argument('--epochs', type=int, default=30, help='training epochs')
parse.add_argument('--patience', type=int, default=2, help='patience of early stopping')
parse.add_argument('--lrate', type=float, default=3e-5, help='learning rate')
parse.add_argument('--beta1', type=float, default=0.9, help='Adam optimizer beta1')
parse.add_argument('--beta2', type=float, default=0.98, help='Adam optimizer beta2')
parse.add_argument('--wdecay', type=float, default=0.01, help='Adam optimizer weight decay')
parse.add_argument('--scheduler', type=str, default="linear", help='scheduler type')
parse.add_argument('--warmup_steps', type=int, default=100_000, help='scheduler warmup steps')
parse.add_argument('--test_split', type=float, default=0.17, help='percentage of train to use as test')
parse.add_argument('--dev_split', type=float, default=0.17, help='percentage of train to use as dev')
parse.add_argument('--seed', type=int, default=32, help='random state of train-test-dev split')


parse.add_argument('--data', type=str, help='data of type csv for classification')
parse.add_argument('--encoding', type=str, default="utf8", help='text encoding')
parse.add_argument('--category', type=str, help='column used for classification')
parse.add_argument('--text', type=str, help='column used for text')
parse.add_argument('--nclasses', type=int, default=0, help='number of classes')
parse.add_argument('--model_path', type=str, help='directory to save model')
parse.add_argument('--checkpoint', type=str, help='path to save the checkpoints')
parse.add_argument('--remove_recessive', type=int, default=3, help='remove classes if they have less than # examples')


args = parse.parse_args()

print()

# Load Pretrained Model
print("Loading Pretrained Model")
tokenizer = None
bert = None
if args.model == 'camembert':
    tokenizer = CamembertTokenizer.from_pretrained(args.pretrained_path)
    bert = CamembertModel.from_pretrained(args.pretrained_path)
elif args.model == 'barthez':
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    bert = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_path).model
elif args.model == 'custom':
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    bert = BertModel.from_pretrained(args.pretrained_path, local_files_only=True)

print()

# Classification Model 
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.cbert = bert
    self.dropout1 = nn.Dropout(p=args.dropout)
    self.linear1 = nn.Linear(self.cbert.config.hidden_size, self.cbert.config.hidden_size)
    self.tanh = nn.Tanh()
    self.dropout2 = nn.Dropout(p=args.dropout)
    self.linear2 = nn.Linear(self.cbert.config.hidden_size, nclasses);
    self.softmax = nn.LogSoftmax(-1)
    
  def forward(self, input_tensor):
    x = self.cbert(input_tensor)[0]
    x = self.dropout1(x)
    x = self.linear1(x)
    x = self.tanh(x)
    x = self.dropout2(x)
    out = self.linear2(x[:,0,:])
    return self.softmax(out)

# Custom Dataset to load legal files
class LegalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings#["input_ids"]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {"input_ids": torch.tensor(self.encodings.iloc[index])}
        item["labels"] = torch.tensor([self.labels.iloc[index]])
        return item



# Early stopping to stop the training when the loss does not improve after certain epochs.
class EarlyStopping():
    def __init__(self, patience=5, tol=0, checkpoints=True):
        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.checkpoints = checkpoints
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            if self.checkpoints: checkpoint(model, tokenizer)
        elif self.best_loss - val_loss > self.tol:
            self.best_loss = val_loss
            self.counter = 0
            if self.checkpoints: checkpoint(model, tokenizer)
        elif self.best_loss - val_loss <= self.tol:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

# Saving and loading models
def save_model(model, tokenizer, model_path):
  torch.save(model.state_dict(), model_path+"/model.pt")
  tokenizer.save_pretrained(model_path)
def load_model(model, model_path):
  model.load_state_dict(torch.load(model_path+"/model.pt"))
  model.eval()
  tokenizer = CamembertTokenizer.from_pretrained(model_path+"/tokenizer_config.json", do_lowercase=False, max_length=312, truncate=True, padding=True, pad_to_max_length=True)
  return model, tokenizer
def checkpoint(model, tokenizer):
  torch.save(model.state_dict(), checkpoints_path+"/model.pt")
  tokenizer.save_pretrained(checkpoints_path)
def load_checkpoint(model):
  model.load_state_dict(torch.load(checkpoints_path+"/model.pt"))
  model.eval()
  tokenizer = CamembertTokenizer.from_pretrained(checkpoints_path+"/tokenizer_config.json", do_lowercase=False, max_length=312, truncate=True, padding=True, pad_to_max_length=True)
  return model, tokenizer


# Collate data used in DataLoader
def collate(data):
    content_sequences = torch.zeros(size=(len(data), sequence_len), dtype=torch.long, device=device)
    for i in range(len(data)):
        sequence = data[i]['input_ids']
        content_sequences[i, :sequence.shape[0]] = sequence
    categories = torch.tensor([el['labels'] for el in data],
                              dtype=torch.long, device=device)
    del data
    return content_sequences, categories

# Plot evolution of loss
def plot_values(train_values, dev_values, label, model_path): 
    plt.plot(train_values, 'b', label='train '+label)
    plt.plot(dev_values, 'r', label='validation '+label)
    plt.ylabel(label)
    plt.xlabel('iterations')
    plt.legend()
    if model_path != None: plt.savefig(f"{model_path}/{label}.png")
    plt.show()


def fit(dataloader, model, loss_fn, optimizer, scheduler, progress_bar=None):
  size = len(dataloader.dataset)
  losses = []
  model.train()
  for i, data in enumerate(dataloader):
    X = data[0].to(device) #data["input_ids"].to(device)
    Y = torch.flatten(data[1]).to(device) #torch.flatten(data["labels"]).to(device)
    pred = model(X)
    loss = loss_fn(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if progress_bar is not None: progress_bar.update(1)
    losses.append(loss.cpu().detach().numpy())
  return np.sum(losses)/size #CPU

def predict(dataloader, model, loss_fn, progress_bar=None, report=False, target_n=None):
  size = len(dataloader.dataset)
  loss = 0
  accuracy = 0
  predictions = []
  targets = []
  model.eval()
  with torch.no_grad():
    for data in dataloader:
      X = data[0].to(device) #data['input_ids'].to(device)
      Y = data[1].to(device) #data['labels'].to(device)
      pred = model(X)
      loss += loss_fn(pred,torch.flatten(Y)).item()
      for idx, probs in enumerate(pred):
        max_prob = torch.argmax(probs, -1)
        accuracy += torch.sum(max_prob == Y[idx]) 
        predictions.append(max_prob.cpu().detach().numpy().item()) 
      if progress_bar is not None: progress_bar.update(1)
      targets.extend(Y.cpu().detach().numpy())
  f1_sample_score = f1_score(targets, predictions, average="micro")
  loss /= size
  if report==True: 
    cr = classification_report(targets, predictions, target_names=target_n, output_dict=False)
    print(cr)
    cr = classification_report(targets, predictions, target_names=target_n, output_dict=True)
    cm = confusion_matrix(label_encoder.inverse_transform(targets), label_encoder.inverse_transform(predictions), labels=target_n)
    print(cm)
    return loss, (accuracy.cpu().detach().numpy())/size, f1_sample_score, cr, cm
  return loss, (accuracy.cpu().detach().numpy())/size, f1_sample_score



def trainModel(model, loaders, epochs, loss_fn, patience, model_path, tokenizer, target_n):
  #Get dataloaders (train,dev,test)
  train_dataloader = loaders[0]
  dev_dataloader = loaders[1]
  test_dataloader = loaders[2]

  num_training_steps = epochs * len(train_dataloader)
  progress_steps = epochs * (2*len(train_dataloader) + len(dev_dataloader))
  progress_bar = tqdm(range(progress_steps))
  
  #Set optimizer options
  optimizer = AdamW(model.parameters(), lr=args.lrate, betas=(args.beta1,args.beta2), eps=1e-6, weight_decay=args.wdecay) 
  lr_scheduler = get_scheduler(
      args.scheduler,
      optimizer=optimizer,
      num_warmup_steps=args.warmup_steps, 
      num_training_steps=num_training_steps
  )
  early_stopping = EarlyStopping(patience)
  
  train_losses = []
  dev_losses = []
  train_accs = []
  dev_accs = []
  train_f1 = []
  dev_f1 = []
  for i in range(epochs):
      print("Epoch %d/%d" %(i+1,epochs))
      loss = fit(train_dataloader, model, loss_fn, optimizer, lr_scheduler, progress_bar)
      print("Loss", loss)
      
      loss, accuracy, f1score = predict(train_dataloader, model, loss_fn, progress_bar)
      train_losses.append(loss)
      train_accs.append(accuracy)
      train_f1.append(f1score)
      print("Train Loss",loss,"Accuracy",accuracy,"F1 score",f1score)
      
      loss, accuracy, f1score = predict(dev_dataloader, model, loss_fn, progress_bar)
      dev_losses.append(loss)
      dev_accs.append(accuracy)
      dev_f1.append(f1score)
      print("Dev Loss",loss,"Accuracy",accuracy,"F1 score",f1score)
      
      early_stopping(loss)
      if early_stopping.early_stop:
          print("Loading model from checkpoint")
          model, tokenizer = load_checkpoint(model)
          break
  #Save the fine-tuned model and the tokenizer
  save_model(model, tokenizer, model_path)
  
  plot_values(train_losses, dev_losses, "loss", model_path)
  plot_values(train_accs, dev_accs, "accuracy", model_path)
  plot_values(train_f1, dev_f1, "f1-score", model_path)

  loss, accuracy, f1score, _, _ = predict(test_dataloader, model.to(device), loss_fn, report=True, target_n=target_n)
  print("Test Loss",loss,"Accuracy",accuracy,"F1 score",f1score)

  return train_losses, dev_losses, train_accs, dev_accs, train_f1, dev_f1


print("Reading CSV")
data = pd.read_csv(args.data, encoding=args.encoding, header=0)
n, p = data.shape
print("The dataframe consists of "+str(n)+" rows and "+ str(p)+" columns.")


# Search for recessive classes
if args.remove_recessive > 0 :
    value_counts = data[args.category].value_counts()
    to_be_removed = value_counts.loc[value_counts < args.remove_recessive].index.values.tolist()
    print('Removing recessive classes:')
    print(to_be_removed)
    for item in to_be_removed:
        data = data.loc[data[args.category] != item]
    print("New shape:", data.shape)


# Label Encoding
print('Encodings Labels')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(data[args.category])
data[args.category] = label_encoder.transform(data[args.category])



# Training Arguments
sequence_len = args.sequence_len
batch_size = args.batch_size
training_epochs = args.epochs
checkpoints_path = args.checkpoint
max_patience = args.patience
if args.nclasses != 0 :
    nclasses = args.nclasses
else:
    targets = list(data[args.category].unique())
    nclasses = len(targets)
model_path = args.model_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories
try: 
  os.mkdir(checkpoints_path)
except OSError:
  print()
try:
  os.mkdir(model_path)
except OSError:
  print()


# Model 
model = Model().to(device)
loss_fn = nn.NLLLoss()

# Spliting Data
X_train, X_test, Y_train, Y_test = train_test_split(data[args.text], data[args.category], test_size=args.test_split, shuffle=True, stratify=data[args.category], random_state=args.seed)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=args.dev_split, shuffle=True, stratify=Y_train, random_state=args.seed)
print("Train dataset has %d examples and %d unique classes" %(X_train.shape[0], len(Y_train.unique()) ) )
print("Dev dataset has %d examples and %d unique classes" %(X_dev.shape[0], len(Y_dev.unique()) ) )
print("Test dataset has %d examples and %d unique classes" %(X_test.shape[0], len(Y_test.unique()) ) )


tokenize = lambda x: tokenizer.encode(x, truncation=True, padding=True, max_length=sequence_len, pad_to_max_length=True)

print("Tokenizing")
print('----train----')
start_time = time.time()
train_encodings = X_train.apply(tokenize)
print("--- %s seconds ---" % (time.time() - start_time))
print('----dev----')
start_time = time.time()
dev_encodings = X_dev.apply(tokenize)
print("--- %s seconds ---" % (time.time() - start_time))
print('----test----')
start_time = time.time()
test_encodings = X_test.apply(tokenize)
print("--- %s seconds ---" % (time.time() - start_time))



print('Creating Datasets')
train_dataset = LegalDataset(train_encodings, Y_train)
dev_dataset = LegalDataset(dev_encodings, Y_dev)
test_dataset = LegalDataset(test_encodings, Y_test)

print('Creating DataLoaders')
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate)
dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate)

dataloaders = [train_dataloader, dev_dataloader, test_dataloader] #train,dev,test


train_l, dev_l, train_a, dev_a, train_f, dev_f = trainModel(model, dataloaders, training_epochs, loss_fn, max_patience, model_path, tokenizer, target_n=list(label_encoder.classes_))

