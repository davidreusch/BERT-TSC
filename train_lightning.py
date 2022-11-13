import matplotlib.pyplot as plt
import typing
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
from IPython.display import display, Markdown, Latex, HTML
from transformers import BertTokenizer, BertModel, BatchEncoding
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from pprint import pprint
from config import cfg
from model_bert import TSCModel_PL


def load_data() -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """load train/validation and test data from csv files into pandas dataframes

    Returns:
        typing.Tuple[pd.DataFrame, pd.DataFrame]: train and validation set, test set
    """
    data_pth = "Data/"
    train_pth = data_pth + "train.csv"
    test_pth = data_pth + "test.csv"
    train_val = pd.read_csv(train_pth)
    test = pd.read_csv(test_pth)
    # print(test)
    return train_val, test


def train_val_split(
    dataset: pd.DataFrame, train_frac: float
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """split dataset according into train and validation set, according to train_frac

    Args:
        dataset (pd.DataFrame): dataset containing input and labels
        train_frac (float): fraction of dataset going to train set

    Returns:
        typing.Tuple[pd.DataFrame, pd.DataFrame]: train and validation set
    """
    out = dataset.copy()
    ind = dataset.index
    perm = np.random.permutation(ind)
    split = int(len(perm) * train_frac)
    perm_train, perm_val = perm[:split], perm[split:]
    train, val = out.iloc[perm_train], out.iloc[perm_val]
    return train, val


def make_tokenized_batches(
    dataset: pd.DataFrame, tokenizer, batch_size: int, device
) -> typing.List[BatchEncoding]:
    """divide the dataset into batches of size batch_size and
    tokenize the batches with the BertTokenizer

    Args:
        dataset (pd.DataFrame): dataset to batch
        tokenizer: tokenizer to use as returned from BertTokenizer.from_pretrained()
        batch_size (int): size of batches

    Returns:
        typing.List[BatchEncoding]: List of dictionarys, one dictionary per batch
    """

    label_tags = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    batches = []
    for b in range(0, len(dataset), batch_size):
        ds_batch = dataset.iloc[b : b + batch_size]
        if len(ds_batch) == 0:
            break
        sentences = ds_batch["comment_text"].to_list()
        token_dict = tokenizer(
            sentences, return_tensors="pt", padding="longest", truncation=True
        )
        labels = ds_batch[label_tags].to_numpy(dtype=int)
        token_dict["labels"] = torch.tensor(labels, dtype=torch.float)
        batches.append(token_dict)
    return batches


class DatasetAdapter(Dataset):
    def __init__(self, batch_encodings) -> None:
        self.batch_encodings = batch_encodings

    def __getitem__(
        self, index
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.batch_encodings[index]["input_ids"],
            self.batch_encodings[index]["token_type_ids"],
            self.batch_encodings[index]["attention_mask"],
            self.batch_encodings[index]["labels"],
        )

    def __len__(self):
        return len(self.batch_encodings)


if __name__ == "__main__":
    cfg = Namespace(**cfg)
    device = (
        torch.device("cuda")
        if False and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(device)
    cfg.device = device

    train_set, test_set = load_data()
    train, val = train_val_split(train_set, train_frac=0.8)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    train_batches = make_tokenized_batches(
        train.iloc[:512], tokenizer, cfg.batchsize, device=device
    )
    val_batches = make_tokenized_batches(
        val.iloc[:512], tokenizer, cfg.batchsize, device=device
    )
    print({k: v.shape for (k, v) in train_batches[0].items()})

    train_loader = DataLoader(DatasetAdapter(train_batches), num_workers=12)
    val_loader = DataLoader(DatasetAdapter(val_batches), num_workers=12)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    cfg.num_training_steps = num_training_steps

    pretrained_model = BertModel.from_pretrained("bert-base-cased")
    pretrained_state_dict = pretrained_model.state_dict()
    model = TSCModel_PL(cfg, pretrained_state_dict)
    trainer = pl.Trainer(gpus=0)
    trainer.fit(model, train_loader, val_loader)
