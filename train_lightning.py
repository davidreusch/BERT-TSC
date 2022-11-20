import matplotlib.pyplot as plt
import os
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
    test_labels_pth = data_pth + "test_labels.csv"
    train_val = pd.read_csv(train_pth)
    test = pd.read_csv(test_pth)

    test_labels = pd.read_csv(test_labels_pth)
    test_labels = test_labels[test_labels.drop(columns=["id"]).sum(axis=1) >= 0]
    test = test.merge(test_labels, on="id")
    return train_val, test


def random_split(
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
    dataset: pd.DataFrame, tokenizer, batch_size: int, tag: str
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

    if os.path.exists(f"Data/{tag}_batches_bs{batch_size}.pt"):
        print("loading batches from disk")
        return torch.load(f"Data/{tag}_batches_bs{batch_size}.pt")
    batches = []
    for b in range(0, len(dataset), batch_size):
        ds_batch = dataset.iloc[b : b + batch_size]
        if len(ds_batch) == 0:
            break
        sentences = ds_batch["comment_text"].to_list()
        token_dict = tokenizer(
            sentences, return_tensors="pt", padding="longest", truncation=True
        )
        labels = ds_batch[cfg.label_tags].to_numpy(dtype=int)
        token_dict["labels"] = torch.tensor(labels, dtype=torch.float)
        batches.append(token_dict)
    torch.save(batches, f"Data/{tag}_batches_bs{batch_size}.pt")
    return batches


class DatasetAdapter(Dataset):
    def __init__(self, batch_encodings) -> None:
        self.batch_encodings = batch_encodings

    def __getitem__(
        self, index
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.batch_encodings[index]["input_ids"],
            self.batch_encodings[index]["token_type_ids"],
            # self.batch_encodings[index]["attention_mask"],
            self.batch_encodings[index]["labels"],
        )

    def __len__(self):
        return len(self.batch_encodings)


if __name__ == "__main__":
    cfg = Namespace(**cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    train_val_set, test_set = load_data()
    # train_set, val_set = random_split(train_val_set, train_frac=0.9)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    train_batches = make_tokenized_batches(
        train_val_set, tokenizer, cfg.batchsize, tag="train"
    )
    # val_batches = make_tokenized_batches(val_set, tokenizer, cfg.batchsize, tag="val")
    val_batches = make_tokenized_batches(test_set, tokenizer, cfg.batchsize, tag="test")
    # train_batches = train_batches[:200]
    # val_batches = val_batches[:100]

    train_loader = DataLoader(DatasetAdapter(train_batches), num_workers=12)
    val_loader = DataLoader(DatasetAdapter(val_batches), num_workers=12)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    cfg.num_training_steps = num_training_steps

    pretrained_model = BertModel.from_pretrained("bert-base-cased")
    pretrained_state_dict = pretrained_model.state_dict()
    model = TSCModel_PL(cfg, pretrained_state_dict)
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs)
    trainer.fit(model, train_loader, val_loader)
