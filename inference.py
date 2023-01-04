from argparse import Namespace

import torch

from config import config_dict

cfg = Namespace(**config_dict)
from transformers import BatchEncoding, BertModel, BertTokenizer

from model_bert import TSCModel_PL
from train_lightning import LazyDatasetAdapter, load_csvs, load_data


def load_data_for_inference(tokenizer):
    train, test = load_csvs()
    test_set = LazyDatasetAdapter(test, tokenizer=tokenizer, batchsize=1)
    return test_set


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # get bert tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    test_data = load_data_for_inference(bert_tokenizer)

    checkpoint_path = "lightning_logs/cluster/cluster3/checkpoints/epoch=0-step=0.ckpt"
    checkpoint = torch.load(checkpoint_path)
    pretrained_state_dict = checkpoint["TSCModel_PL"]

    # load model with pretrained weights
    model = TSCModel_PL(cfg, pretrained_state_dict, inverse_class_probabilities=[])

    # go on to do the inference loop
    outputs = []
    for i, batch in enumerate(test_data):
        input_ids, token_type_ids, labels = batch
        batchsize, seq_len = input_ids.shape
        position_ids = torch.stack(
            [torch.arange(0, seq_len, dtype=torch.long, device=device)] * batchsize
        )
        prediction = model(input_ids, token_type_ids, position_ids)

        sentence = test_data.dataset.df.iloc[i]["comment_text"]
        outputs.append((prediction, labels, sentence))


if __name__ == "__main__":
    main()
