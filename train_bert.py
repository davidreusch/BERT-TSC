import matplotlib.pyplot as plt
import typing
import random
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from IPython.display import display, Markdown, Latex, HTML
from transformers import BertTokenizer, BertModel, BatchEncoding
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from pprint import pprint
from config import cfg, label_tags
from model_bert import ToxicSentimentClassificationModel


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
    dataset: pd.DataFrame, tokenizer, batch_size: int
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


def get_accuracy(logits: torch.Tensor, labels: torch.Tensor, verbose=False):
    if verbose:
        nonzero_labels = labels[labels.sum(dim=-1) > 0]
        nonzero_logits = logits[labels.sum(dim=-1) > 0]
        for i in range(nonzero_labels.shape[0]):
            pprint(torch.sigmoid(nonzero_logits)[i])
            pprint(nonzero_labels[i])
    predictions = (torch.sigmoid(logits) > 0.5).type(torch.float)

    errors = torch.mean(torch.abs(predictions - labels))
    return 1 - errors


def analyse_data(dataset: pd.DataFrame):

    labels = dataset[label_tags].to_numpy(dtype=float)
    print(labels)
    print(labels.shape)
    a = labels.sum(axis=-1) / 6
    nonzero_rows = a[a != 0]
    print(nonzero_rows)
    print(len(nonzero_rows))

    print(
        "ratio of nonzero-labels in dataset:", labels.mean()
    )  # ~ 0.0367 so 1 / 27 of the dataset


if __name__ == "__main__":
    cfg = Namespace(**cfg)

    train_set, test_set = load_data()

    train, val = random_split(train_set, train_frac=0.8)
    analyse_data(train_set)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    train_batches = make_tokenized_batches(train.iloc[:1024], tokenizer, cfg.batchsize)
    val_batches = make_tokenized_batches(val.iloc[:512], tokenizer, cfg.batchsize)

    print(train_batches[0])

    device = (
        torch.device("cuda")
        if False and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(device)
    cfg.device = device
    pretrained_model = BertModel.from_pretrained("bert-base-cased")
    pretrained_state_dict = pretrained_model.state_dict()
    # pprint({k: v.shape for (k, v) in pre_state_dict.items()})
    # pre_state_dict_adapted = {'backbone.' + k: v for (k, v) in pre_state_dict.items()}
    # pprint({k: v.shape for (k, v) in pre_state_dict_adapted.items()})
    model = ToxicSentimentClassificationModel(cfg, pretrained_state_dict)
    model.to(device)
    print()

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_batches)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)

    progress_bar = tqdm(range(num_training_steps))

    positive_weights = 27 * torch.ones(cfg.num_target_categories)
    loss_fn = nn.BCEWithLogitsLoss(positive_weights)

    for epoch in range(num_epochs):
        model.train()
        print("epoch:", epoch)
        train_losses = []
        random.shuffle(train_batches)
        for batch in train_batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs, batch["labels"])
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            accuracy = get_accuracy(outputs, batch["labels"], verbose=True)
            progress_bar.update(1)
            print("loss:", loss.item())
            print("accuracy:", accuracy.item())
            train_losses.append(loss.item())

        print("mean train loss for epoch", sum(train_losses) / len(train_losses))
        model.eval()
        val_losses = []
        for batch in val_batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs, batch["labels"])
            accuracy = get_accuracy(outputs, batch["labels"])
            print("loss:", loss.item())
            print("accuracy:", accuracy.item())
            val_losses.append(loss.item())
        print("mean validation loss for epoch", sum(val_losses) / len(val_losses))
