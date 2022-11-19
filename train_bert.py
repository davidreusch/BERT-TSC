import matplotlib.pyplot as plt
import os
import typing
import random
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
from IPython.display import display, Markdown, Latex, HTML
from transformers import BertTokenizer, BertModel, BatchEncoding
from transformers import get_scheduler
from tqdm.auto import tqdm
from pprint import pprint
from config import cfg, label_tags
from model_bert import ToxicSentimentClassificationModel
from sklearn.metrics import roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay


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
        return torch.load(f"Data/{tag}_batches.pt")
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
    torch.save(batches, f"Data/{tag}_batches_bs{batch_size}.pt")
    return batches


def get_accuracy(pred_probs: torch.tensor, labels: torch.tensor, verbose=False):
    with torch.no_grad():
        if verbose:
            nonzero_labels = labels[labels.sum(dim=-1) > 0]
            nonzero_logits = pred_probs[labels.sum(dim=-1) > 0]
            for i in range(nonzero_labels.shape[0]):
                pprint(nonzero_logits)[i]
                pprint(nonzero_labels[i])
        predictions = (pred_probs > 0.5).float()
        acc = (predictions == labels).float().mean()
    return acc


def get_roc(labels: np.ndarray, pred_probs: np.ndarray, tag=""):
    non_zero_cols = labels.sum(axis=0) > 0
    non_zero_labels = non_zero_cols.nonzero()[0]

    roc_auc = roc_auc_score(
        labels[:, non_zero_cols], pred_probs[:, non_zero_cols], average=None
    )

    if tag:
        for label in non_zero_labels:
            ConfusionMatrixDisplay.from_predictions(
                labels[:, label].astype(int), (pred_probs[:, label] > 0.5)
            )
            plt.savefig(f"plots/confusion_matrix_{tag}_{label_tags[label]}.png")
            plt.cla()
            RocCurveDisplay.from_predictions(
                labels[:, label].astype(int), pred_probs[:, label]
            )
            plt.savefig(f"plots/roc_curve_{tag}_{label_tags[label]}.png")
            plt.cla()
            plt.close()

    print("ROC for labels:\n", non_zero_labels.tolist())
    print(roc_auc)
    return sum(roc_auc) / len(roc_auc)


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


def validate(
    model: nn.Module,
    batches: typing.List[BatchEncoding],
    device: torch.device,
    epoch: int,
    verbose: bool = False,
):
    model.eval()
    positive_weights = 27 * torch.ones(cfg.num_target_categories, device=device)
    loss_fn = nn.BCEWithLogitsLoss(positive_weights)
    loss_sum = 0
    acc_sum = 0
    pred_probs = []
    with torch.no_grad():
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            logits = model(input_ids, token_type_ids)
            labels = batch["labels"].to(device)
            loss_sum += loss_fn(logits, labels).item()
            pred_prob = torch.sigmoid(logits.detach()).cpu()
            pred_probs.append(pred_prob)
            acc_sum += get_accuracy(pred_prob, labels.detach().cpu(), verbose=verbose)
        pred_probs = np.concatenate(pred_probs, axis=0)
        val_labels = (
            torch.cat([batch["labels"] for batch in batches]).detach().cpu().numpy()
        )
    return (
        loss_sum / len(batches),
        acc_sum / len(batches),
        get_roc(val_labels, pred_probs, tag=f"val_epoch_{epoch}"),
    )


def train(
    model: nn.Module,
    train_batches: typing.List[BatchEncoding],
    val_batches: typing.List[BatchEncoding],
    device: torch.device,
    cfg: Namespace,
    log_file: str,
):
    model.train()
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=cfg.num_epochs * len(train_batches),
    )
    positive_weights = 27 * torch.ones(cfg.num_target_categories, device=device)
    loss_fn = nn.BCEWithLogitsLoss(positive_weights)

    for epoch in range(cfg.num_epochs):
        loss_sum = 0
        acc_sum = 0
        random.shuffle(train_batches)
        pred_probs = []
        for i, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            logits = model(input_ids, token_type_ids)
            labels = batch["labels"].to(device)
            loss_val = loss_fn(logits, labels)

            loss_val.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss_val.item()
            pred_prob = torch.sigmoid(logits.detach()).cpu()
            pred_probs.append(pred_prob)
            acc_sum += get_accuracy(pred_prob, labels.detach().cpu())
            if i % cfg.log_interval == 0:
                print(
                    f"Epoch {epoch}, batch {i}: loss {loss_sum / (i + 1)}, acc {acc_sum / (i + 1)}"
                )
        train_loss = loss_sum / len(train_batches)
        train_acc = acc_sum / len(train_batches)
        pred_probs = np.concatenate(pred_probs, axis=0)
        train_labels = torch.cat([batch["labels"] for batch in train_batches])
        train_labels = train_labels.detach().cpu().numpy()
        train_roc = get_roc(train_labels, pred_probs, tag=f"train_epoch_{epoch}")
        val_loss, val_acc, val_roc = validate(model, val_batches, device, epoch)
        print(
            f"epoch {epoch}: train loss {train_loss}, train acc {train_acc}, train roc {train_roc}, val loss {val_loss}, val acc {val_acc}, val roc {val_roc}"
        )
        # torch.save(model.state_dict(), f"models/model_{epoch}.pt")


if __name__ == "__main__":
    cfg = Namespace(**cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    cfg.device = device

    train_val_set, test_set = load_data()
    # analyse_data(train_val_set)

    train_set, val_set = random_split(train_val_set, train_frac=0.8)
    print(train_set)
    print(val_set)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    train_batches = make_tokenized_batches(
        train_set, tokenizer, cfg.batchsize, tag="train"
    )
    val_batches = make_tokenized_batches(val_set, tokenizer, cfg.batchsize, tag="val")

    pretrained_model = BertModel.from_pretrained("bert-base-cased")
    pretrained_state_dict = pretrained_model.state_dict()
    model = ToxicSentimentClassificationModel(cfg, pretrained_state_dict)
    model.to(device)

    train(model, train_batches, val_batches, device, cfg, "log.txt")
