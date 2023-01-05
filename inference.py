from argparse import Namespace

import numpy as np
import pandas as pd
import torch

from config import config_dict

cfg = Namespace(**config_dict)
from transformers import BertTokenizer

from model_bert import TSCModel_PL
from train_apply import LazyDatasetAdapter, load_csvs, load_data


def load_data_for_inference(tokenizer):
    train, test = load_csvs()
    test_set = LazyDatasetAdapter(test, tokenizer=tokenizer, batchsize=1)
    return test_set


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # get bert tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # load data (only train and test, no validation, because all that matters in the end is performance on test set)
    test_data = load_data_for_inference(bert_tokenizer)

    _, _, inverse_class_probabilities = load_data(
        dataset="jigsaw-TSC",
        transformation=bert_tokenizer,
        batchsize=cfg.batchsize,
        early_loading=True,
    )

    checkpoint_path = "lightning_logs/version_22/checkpoints/epoch=1-step=9974.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    checkpoint_state_dict = checkpoint["state_dict"]
    pretrained_state_dict = {
        k.replace("backbone.", ""): v for k, v in checkpoint_state_dict.items()
    }

    TSC_state_dict = {
        "output_layer.linear.weight": pretrained_state_dict["output_layer.linear.weight"],
        "output_layer.linear.bias": pretrained_state_dict["output_layer.linear.bias"],
    }
    pretrained_state_dict.popitem()
    pretrained_state_dict.popitem()

    # load BERT model with pretrained weights
    model = TSCModel_PL(cfg, pretrained_state_dict, inverse_class_probabilities)

    # go on to do the inference loop
    model.eval()
    with torch.no_grad():
        # load output layer with pretrained weights
        # w = torch.Tensor(TSC_state_dict).reshape(model.weight.shape)
        model.output_layer.linear.weight.copy_(TSC_state_dict["output_layer.linear.weight"])
        model.output_layer.linear.bias.copy_(TSC_state_dict["output_layer.linear.bias"])

        outputs = []
        for i, batch in enumerate(test_data):
            input_ids, token_type_ids, labels = batch
            batchsize, seq_len = input_ids.shape
            print("sample:", i)

            position_ids = torch.stack(
                [torch.arange(0, seq_len, dtype=torch.long, device=device)] * batchsize
            )

            logits = model(input_ids, token_type_ids, position_ids)
            prediction = torch.sigmoid(logits)

            sentence = test_data.dataset.iloc[i]["comment_text"]

            outputs.append((sentence, np.array(labels), np.array(prediction)))

            if i == 63976:
                break

        pd.DataFrame(outputs, columns=["comment_text", "labels", "prediction"]).to_csv(
            "Data/outputs.csv"
        )
        print("Outputs saved")


if __name__ == "__main__":
    main()
