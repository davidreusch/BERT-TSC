from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from torch.optim import AdamW
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelConfusionMatrix,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelROC,
)
from transformers import get_linear_schedule_with_warmup

import utils


class BertSelfAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.query = nn.Linear(cfg.d_model, cfg.d_model)
        self.key = nn.Linear(cfg.d_model, cfg.d_model)
        self.value = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.p_dropout)
        self.num_attention_heads = cfg.num_attention_heads

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq.shape = (batchsize, seq_len, d_model)
        batchsize, seq_len, d_model = seq.shape
        assert d_model % self.num_attention_heads == 0
        d_v = d_model // self.num_attention_heads
        querys_proj = self.query(seq).view(batchsize, seq_len, self.num_attention_heads, d_v)
        querys_proj = querys_proj.transpose(1, 2)
        keys_proj = self.key(seq).view(batchsize, seq_len, self.num_attention_heads, d_v)
        keys_proj = keys_proj.transpose(1, 2)
        values_proj = self.value(seq).view(batchsize, seq_len, self.num_attention_heads, d_v)
        values_proj = values_proj.transpose(1, 2)

        #   (batchsize, num_attention_heads, seq_len, d_v) x (batchsize, num_attention_heads, d_v, seq_len)
        # = (batchsize, num_attention_heads, seq_len, seq_len)
        A = querys_proj @ keys_proj.transpose(-1, -2) / d_v**0.5
        p_attention = nn.Softmax(dim=-1)(A)
        #   (batchsize, num_attention_heads, seq_len, seq_len) x (batchsize, num_attention_heads, seq_len, d_v)
        # = (batchsize, num_attention_heads, seq_len, d_v)
        attention_output = p_attention @ values_proj

        # concatenate the heads along the last axis by reshaping the output to (batchsize, seq_len, d_model) again
        attention_output = attention_output.transpose(1, 2).reshape(
            batchsize, seq_len, d_v * self.num_attention_heads
        )
        return attention_output


class BertSelfOutput(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.LayerNorm = nn.LayerNorm(cfg.d_model, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(cfg.p_dropout)

    def forward(self, attention_output: torch.Tensor) -> torch.Tensor:
        # attention_output.shape = (batchsize, seq_len, d_model)
        # in which order to apply these things? see below
        return self.dropout(self.LayerNorm(self.dense(attention_output)))


class BertAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.self = BertSelfAttention(cfg)
        self.output = BertSelfOutput(cfg)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        attention_output = self.self(seq)
        return self.output(attention_output)


class BertIntermediate(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.d_model, cfg.d_model * 4)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(seq))


class BertOutput(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.d_model * 4, cfg.d_model)
        self.LayerNorm = nn.LayerNorm(cfg.d_model, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(cfg.p_dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # in which order to apply dropout, layernorm and residual connection?
        # in Vaswani et al it is layernorm(x + sublayer(x))
        # the order here is from annotated transformer
        return self.dropout(self.LayerNorm(self.dense(seq)))


class BertLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.attention = BertAttention(cfg)
        self.intermediate = BertIntermediate(cfg)
        self.output = BertOutput(cfg)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(seq) + seq
        x = self.intermediate(attention_output)
        x = self.output(x) + attention_output
        return x


class BertEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        encoder_stack = [BertLayer(cfg) for _ in range(cfg.num_encoder_blocks)]
        self.layer = nn.ModuleList(encoder_stack)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        encoder_stack = nn.Sequential(*self.layer)
        return encoder_stack(seq)


class BertEmbeddings(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embeddings = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.token_type_embeddings = nn.Embedding(
            2, cfg.d_model
        )  # only beginning of sentence token and other tokens
        self.LayerNorm = nn.LayerNorm(cfg.d_model, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(cfg.p_dropout)
        self.register_buffer("position_ids", torch.arange(cfg.max_seq_len).expand((1, -1)))

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeds = input_embeds + position_embeds + token_type_embeds
        embeds = self.LayerNorm(input_embeds)
        embeds = self.dropout(embeds)
        return embeds


class BertPooler(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.d_model, cfg.d_model)


class MyBertModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)
        self.pooler = BertPooler(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_output = self.encoder(embeds)
        return encoder_output


class OutputLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.linear = nn.Linear(cfg.d_model, cfg.num_target_categories)
        # do not use neural network but linear layer for classification
        # self.dense1 = nn.Linear(cfg.d_model, cfg.d_model // 2)
        # self.act_fn = nn.GELU()
        # self.dense2 = nn.Linear(cfg.d_model // 2, cfg.num_target_categories)
        # self.dropout = nn.Dropout(cfg.p_dropout)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # encoder_output.shape: (batchsize, seq_len, d_model)
        x = self.linear(encoder_output[:, 0])
        # do not use neural network but linear layer for classification
        # x = self.dense1(
        # encoder_output[:, 0]
        # )  # use only hidden state corresponding to start of sequence token for classification
        # x = self.act_fn(x)
        # x = self.dropout(x)
        # x = self.dense2(x)
        return x


class ToxicSentimentClassificationModel(nn.Module):
    def __init__(self, cfg, state_dict) -> None:
        super().__init__()
        self.backbone = MyBertModel(cfg)
        self.backbone.load_state_dict(state_dict)
        self.output_layer = OutputLayer(cfg)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        backbone_output = self.backbone(input_ids, token_type_ids, position_ids)
        return self.output_layer(backbone_output)


class TSCModel_PL(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = MyBertModel(cfg)
        self.output_layer = OutputLayer(cfg)
        self.cfg = cfg

    def init_train(self, inverse_class_probabilities, warmup_steps, total_steps):
        # code for weighting of loss function
        if inverse_class_probabilities:
            class_weights = torch.tensor(
                [inverse_class_probabilities[tag] for tag in self.cfg.label_tags],
                requires_grad=False,
            )
        else:
            class_weights = torch.ones(len(self.cfg.label_tags), requires_grad=False)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.automatic_optimization = False
        self._outputs = {"train": [], "test": []}
        self.warmup_steps, self.total_steps = warmup_steps, total_steps

        metrics = MetricCollection(
            [
                MultilabelAccuracy(num_labels=6, average=None),
                MultilabelAUROC(num_labels=6, average=None),
                MultilabelPrecision(num_labels=6, average=None),
                MultilabelRecall(num_labels=6, average=None),
            ]
        )

        self.train_mean_loss = MeanMetric()
        self.train_metrics = metrics.clone()
        self.train_roc = MultilabelROC(num_labels=6)
        self.train_confusion_matrix = MultilabelConfusionMatrix(num_labels=6)
        self.test_mean_loss = MeanMetric()
        self.test_metrics = metrics.clone()
        self.test_roc = MultilabelROC(num_labels=6)
        self.test_confusion_matrix = MultilabelConfusionMatrix(num_labels=6)
        self.test_predictions = []
        self.vars = dict(self.__dict__["_modules"])
        super().train(mode=True)

    def load_pretrained_weights_backbone(self, state_dict):
        self.backbone.load_state_dict(state_dict)
        # to freeze backbone weights
        # self.backbone = self.backbone.requires_grad_(False)

    def load_pretrained_weights_whole_model(self, state_dict):
        if "loss_fn.pos_weight" in state_dict:
            del state_dict["loss_fn.pos_weight"]
        self.load_state_dict(state_dict)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        backbone_output = self.backbone(input_ids, token_type_ids, position_ids)
        return self.output_layer(backbone_output)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_with_weight_decay = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params_with_weight_decay, lr=self.cfg.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return [optimizer], [scheduler]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        input_ids, token_type_ids, labels = batch
        batchsize, seq_len = input_ids.shape
        position_ids = torch.stack(
            [torch.arange(0, seq_len, dtype=torch.long, device=self.device)] * batchsize
        )
        logits = self(input_ids, token_type_ids, position_ids=position_ids)
        loss = self.loss_fn(logits, labels) / self.cfg.opt_step_interval
        loss.backward()

        # accumulate gradients of multiple batches
        if (batch_idx + 1) % self.cfg.opt_step_interval == 0:
            opt.step()  # type: ignore
            scheduler.step()  # type: ignore
            opt.zero_grad()  # type: ignore

        logits_cloned = logits.clone().detach()
        labels_cloned = labels.clone().detach()
        self._outputs["train"].append((logits_cloned, labels_cloned))
        self.log_on_step(logits_cloned, labels_cloned, tag="train")
        if (batch_idx + 1) % self.cfg.log_interval == 0:
            self.log_on_interval(tag="train")

    def training_epoch_end(self, outputs):
        self._epoch_end(tag="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        with torch.no_grad():
            input_ids, token_type_ids, labels = batch
            batchsize, seq_len = input_ids.shape
            position_ids = torch.stack(
                [torch.arange(0, seq_len, dtype=torch.long, device=self.device)] * batchsize
            )
            logits = self(input_ids, token_type_ids, position_ids)

            self._outputs["test"].append((logits, labels))
            self.log_on_step(logits, labels, tag="test")
            if (batch_idx + 1) % self.cfg.log_interval == 0:
                self.log_on_interval(tag="test")
        return logits, labels

    def validation_epoch_end(self, outputs):
        self.test_predictions = self._outputs["test"]
        self._epoch_end(tag="test")

    def log_on_interval(self, tag: str):
        logits = torch.cat([o[0] for o in self._outputs[tag]])
        labels = torch.cat([o[1] for o in self._outputs[tag]])
        self.vars[tag + "_metrics"].update(logits, labels.int())
        self.log(
            f"{tag}_accuracy",
            self.vars[tag + "_metrics"].compute()["MultilabelAccuracy"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{tag}_precision",
            self.vars[tag + "_metrics"].compute()["MultilabelPrecision"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{tag}_recall",
            self.vars[tag + "_metrics"].compute()["MultilabelRecall"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.vars[tag + "_metrics"].reset()

    def log_on_step(self, logits, labels, tag="train"):
        self.vars[tag + "_mean_loss"].update(self.loss_fn(logits, labels).item())
        self.log(
            f"{tag}_loss",
            self.vars[tag + "_mean_loss"].compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def _epoch_end(self, tag: str):

        logits = torch.cat([o[0] for o in self._outputs[tag]])
        labels = torch.cat([o[1] for o in self._outputs[tag]])
        self.vars[tag + "_metrics"].update(logits, labels.int())
        metric_dict = self.vars[tag + "_metrics"].compute()
        auroc = metric_dict["MultilabelAUROC"]
        self.vars[tag + "_confusion_matrix"].update(logits, labels.int())
        cm = self.vars[tag + "_confusion_matrix"].compute()
        self.vars[tag + "_roc"].update(logits, labels.int())
        fpr, tpr, thresholds = self.vars[tag + "_roc"].compute()

        tensorboard = self.logger.experiment  # type: ignore
        utils.log_roc_curve(
            fpr,
            tpr,
            tensorboard,
            self.current_epoch,
            tag=tag,
            auroc=auroc,
        )
        utils.log_confusion_matrix(cm, tensorboard, self.current_epoch, tag=tag)
        utils.log_metrics_table(metric_dict, tensorboard, self.current_epoch, tag=tag)
        # self.log_roc_and_confusion_matrix_sklearn(
        # tensorboard, logits.cpu().numpy(), labels.cpu().numpy(), tag=tag
        # )
        self.vars[tag + "_metrics"].reset()
        self.vars[tag + "_confusion_matrix"].reset()
        self.vars[tag + "_roc"].reset()
        self.vars[tag + "_mean_loss"].reset()
        self._outputs[tag] = []

    def log_roc_and_confusion_matrix_sklearn(
        self, tensorboard, pred_probs: np.ndarray, labels: np.ndarray, tag: str
    ):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        non_zero_cols = labels.sum(axis=0) > 0
        non_zero_labels = non_zero_cols.nonzero()[0]
        if len(non_zero_labels) == 0:
            print(f"WARNING: no nonzero labels for {tag}, cannot compute roc")
            return
        roc_auc = roc_auc_score(
            labels[:, non_zero_cols], pred_probs[:, non_zero_cols], average=None
        )
        print("ROC for labels:\n", non_zero_labels.tolist())
        print(roc_auc)

        for l in non_zero_labels:
            label = self.cfg.label_tags[l]
            fig = ConfusionMatrixDisplay.from_predictions(
                labels[:, l].astype(int), (sigmoid(pred_probs[:, l]) > 0.5)
            ).figure_
            tensorboard.add_figure(
                f"confusion_matrix_sklearn_{tag}/{label}",
                fig,
                global_step=self.current_epoch,
                walltime=None,
            )
            fig = RocCurveDisplay.from_predictions(
                labels[:, l].astype(int), pred_probs[:, l]
            ).figure_
            tensorboard.add_figure(
                f"roc_curve_sklearn_{tag}/{label}",
                fig,
                global_step=self.current_epoch,
                walltime=None,
            )

        return sum(roc_auc) / len(roc_auc)

    def get_test_predictions(self) -> np.ndarray:
        if len(self.test_predictions) == 0:
            raise ValueError("No test predictions available")
        return torch.cat([o[0] for o in self.test_predictions]).cpu().numpy()
