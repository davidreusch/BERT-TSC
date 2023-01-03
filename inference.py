from argparse import Namespace

import torch

from config import config_dict

cfg = Namespace(**config_dict)
from transformers import BatchEncoding, BertModel, BertTokenizer

from model_bert import TSCModel_PL
from train_lightning import load_data


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # get bert tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # load data (only train and test, no validation, because all that matters in the end is performance on test set)
    train_loader, test_loader, inverse_class_probabilities = load_data(
        dataset="jigsaw-TSC",
        transformation=bert_tokenizer,
        batchsize=cfg.batchsize,
        early_loading=True,
    )

    checkpoint_path = "lightning_logs/cluster/cluster3/checkpoints/epoch=0-step=0.ckpt"
    checkpoint = torch.load(checkpoint_path)
    pretrained_state_dict = checkpoint["TSCModel_PL"]

    # load model with pretrained weights
    model = TSCModel_PL(cfg, pretrained_state_dict, inverse_class_probabilities)

    # go on to do the inference loop


if __name__ == "__main__":
    main()
