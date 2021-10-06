import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
# tqdm.pandas()


class LegalDataset(Dataset):
  def __init__(self, text):
    self.encodings = text

  def __len__(self):
    return len(self.encodings)

  def __getitem__(self, index):
    item = {"input_ids": torch.tensor(self.encodings.iloc[index])}
    return item


def process_text(filename, name, map_tokenize, encoding):
    print("Opening file...")
    file = open(filename, "r", encoding=encoding)
    text = file.readlines() # list
    file.close()
    text = pd.Series(text)
    tqdm.pandas(desc="Tokenizing")
    text = text.progress_map(map_tokenize)
    dataset = LegalDataset(text)
    text = None
    occ = filename.rfind("/") + 1
    path = filename[:occ]
    torch.save(dataset, path+name+".pt")
    return path+name+".pt"