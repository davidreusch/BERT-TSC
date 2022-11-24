#!/usr/bin/env python3
# $ -N pml_08_bert1 # name of the experiment
# $ -l cuda=1 # remove this line when no GPU is needed!
# $ -q all.q # do not fill the qlogin queue
# $ -cwd # start processes in current directory
# $ -V   # provide environment variables


import sys

sys.path.append("/home/pml_08/BERT-TSC")
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
from transformers import BertTokenizer, BertModel, BatchEncoding
from config import cfg
from model_bert import TSCModel_PL


def load_csvs() -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """load train/validation and test data from csv files into pandas dataframes

    Returns:
        typing.Tuple[pd.DataFrame, pd.DataFrame]: train and test set
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
        not used at the moment

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


def load_data(
    dataset: str = "jigsaw-TSC",
    transformation=None,
    early_loading=False,
    batchsize=4,
) -> typing.Tuple[DataLoader, DataLoader]:
    """either first compute all tokenized batches and load them in ram, return dataloader to load them
        or return dataloader that tokenizes batches on the fly (slower during training)

    Returns:
        train and test dataloader
    """
    if dataset != "jigsaw-TSC":
        raise ValueError("only jigsaw-TSC supported as dataset")
    # load_csvs loads the csvs into dataframes at once
    # we could in principle also read the csvs lazy (in chunks), but with a size of ~60MB per csv,
    # we consider this not necessary
    train, test = load_csvs()
    if early_loading:
        train_batches = make_tokenized_batches(
            train, tokenizer=transformation, batch_size=batchsize, tag="train"
        )
        test_batches = make_tokenized_batches(
            test, tokenizer=transformation, batch_size=batchsize, tag="test"
        )
        train_loader = DataLoader(DatasetAdapter(train_batches), num_workers=12)
        test_loader = DataLoader(DatasetAdapter(test_batches), num_workers=12)
    else:
        train_loader = DataLoader(
            LazyDatasetAdapter(train, transformation, batchsize), num_workers=12
        )
        test_loader = DataLoader(
            LazyDatasetAdapter(test, transformation, batchsize), num_workers=12
        )

    return train_loader, test_loader


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
        # get batch of size batch size from dataset
        ds_batch = dataset.iloc[b : b + batch_size]
        sentences = ds_batch["comment_text"].to_list()
        # tokenize the batch
        token_dict = tokenizer(
            sentences, return_tensors="pt", padding="longest", truncation=True
        )
        # add labels to token_dict
        labels = ds_batch[cfg.label_tags].to_numpy(dtype=int)
        token_dict["labels"] = torch.tensor(labels, dtype=torch.float)
        batches.append(token_dict)
    torch.save(batches, f"Data/{tag}_batches_bs{batch_size}.pt")
    return batches


class DatasetAdapter(Dataset):
    """Adapter class to get out tensors from the tokenized batches
    so that DataLoader is happy
    """

    def __init__(self, batch_encodings) -> None:
        self.batch_encodings = batch_encodings

    def __getitem__(
        self, index
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.batch_encodings[index]["input_ids"],
            self.batch_encodings[index]["token_type_ids"],
            self.batch_encodings[index]["labels"],
        )

    def __len__(self):
        return len(self.batch_encodings)


class LazyDatasetAdapter(Dataset):
    """Adapter class to get out tensors from the tokenized batches, so that DataLoader is happy,
    batches are created on the fly from the raw input dataset, this is much slower than precomputing but scalable to larger datasets
    A better solution would then be to store the precomputed batches in several files on disk and load them as needed or to store them in a database
    """

    def __init__(self, dataset, tokenizer, batchsize) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batchsize = batchsize

    def __getitem__(
        self, i: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """get a batch of size batchsize from the dataset, tokenize it and return the tensors"""
        ds_batch = self.dataset.iloc[i * self.batchsize : (i + 1) * self.batchsize]
        sentences = ds_batch["comment_text"].to_list()
        token_dict = self.tokenizer(
            sentences, return_tensors="pt", padding="longest", truncation=True
        )
        labels = torch.tensor(ds_batch[cfg.label_tags].to_numpy(dtype=float))
        return (
            token_dict["input_ids"],
            token_dict["token_type_ids"],
            labels,
        )

    def __len__(self):
        return len(self.dataset) // self.batchsize + int(
            len(self.dataset) % self.batchsize > 0
        )


if __name__ == "__main__":
    cfg = Namespace(**cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # get bert tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # load data (only train and test, no validation, because all that matters in the end is performance on test set)
    train_loader, test_loader = load_data(
        dataset="jigsaw-TSC",
        transformation=bert_tokenizer,
        batchsize=cfg.batchsize,
        early_loading=True,
    )
    num_training_steps = cfg.num_epochs * len(train_loader)
    cfg.num_training_steps = num_training_steps

    # get pretrained weights of bert
    pretrained_model = BertModel.from_pretrained("bert-base-cased")
    pretrained_state_dict = pretrained_model.state_dict()

    # load model with pretrained weights
    model = TSCModel_PL(cfg, pretrained_state_dict)

    # train the model
    trainer = pl.Trainer(gpus=0, max_epochs=cfg.num_epochs)
    trainer.fit(model, train_loader, test_loader)
