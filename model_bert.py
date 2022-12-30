import pytorch_lightning as pl
import torch
import torch.nn as nn
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

import utils


class BertSelfAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.query = nn.Linear(cfg.d_model, cfg.d_model)
        self.key = nn.Linear(cfg.d_model, cfg.d_model)
        self.value = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.p_dropout)
        self.num_attention_heads = cfg.num_attention_heads

    def forward(self, seq: torch.Tensor):
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
        Z = querys_proj @ keys_proj.transpose(-1, -2) / d_v**0.5
        p_attention = nn.Softmax(dim=-1)(Z)
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

    def forward(self, attention_output):
        # attention_output.shape = (batchsize, seq_len, d_model)
        # in which order to apply these things? see below
        return self.dropout(self.LayerNorm(self.dense(attention_output))) + attention_output


class BertAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.self = BertSelfAttention(cfg)
        self.output = BertSelfOutput(cfg)

    def forward(self, seq):
        attention_output = self.self(seq)
        return self.output(attention_output)


class BertIntermediate(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.d_model, cfg.d_model * 4)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, seq):
        return self.intermediate_act_fn(self.dense(seq))


class BertOutput(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.d_model * 4, cfg.d_model)
        self.LayerNorm = nn.LayerNorm(cfg.d_model, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(cfg.p_dropout)

    def forward(self, seq):
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

    def forward(self, seq):
        attention_output = self.attention(seq)
        x = self.intermediate(attention_output)
        x = self.output(x)
        return x + attention_output


class BertEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        encoder_stack = [BertLayer(cfg) for _ in range(cfg.num_encoder_blocks)]
        self.layer = nn.ModuleList(encoder_stack)

    def forward(self, seq: torch.Tensor):
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
    ):
        # batchsize, seq_len = input_ids.shape
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
    ):
        embeds = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_output = self.encoder(embeds)
        return encoder_output


class OutputLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.dense1 = nn.Linear(cfg.d_model, cfg.d_model // 2)
        self.act_fn = nn.GELU()
        self.dense2 = nn.Linear(cfg.d_model // 2, cfg.num_target_categories)
        self.dropout = nn.Dropout(cfg.p_dropout)

    def forward(self, encoder_output: torch.Tensor):
        # encoder_output.shape: (batchsize, seq_len, d_model)
        x = self.dense1(
            encoder_output[:, 0]
        )  # use only hidden state corresponding to start of sequence token for classification
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.dense2(x)
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
    ):
        backbone_output = self.backbone(input_ids, token_type_ids, position_ids)
        return self.output_layer(backbone_output)


class TSCModel_PL(pl.LightningModule):
    def __init__(self, cfg, state_dict, inverse_class_probabilities) -> None:
        super().__init__()
        self.backbone = MyBertModel(cfg)
        self.backbone.load_state_dict(state_dict)
        self.backbone = self.backbone.requires_grad_(False)

        self.output_layer = OutputLayer(cfg)
        # positive_weights = 27 * torch.ones(cfg.num_target_categories)
        class_weights = torch.tensor(
            [inverse_class_probabilities[tag] for tag in cfg.label_tags]
        )
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.cfg = cfg
        self.automatic_optimization = False
        self._outputs = {"train": [], "val": []}

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
        self.val_mean_loss = MeanMetric()
        self.val_metrics = metrics.clone()
        self.val_roc = MultilabelROC(num_labels=6)
        self.val_confusion_matrix = MultilabelConfusionMatrix(num_labels=6)
        self.vars = dict(self.__dict__["_modules"])

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ):
        backbone_output = self.backbone(input_ids, token_type_ids, position_ids)
        return self.output_layer(backbone_output)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        input_ids, token_type_ids, labels = batch
        batchsize, seq_len = input_ids.shape
        position_ids = torch.stack(
            [torch.arange(0, seq_len, dtype=torch.long, device=self.device)] * batchsize
        )
        logits = self(input_ids, token_type_ids, position_ids=position_ids)
        loss_val = self.loss_fn(logits, labels)
        loss_val.backward()

        # accumulate gradients of multiple batches
        if (batch_idx + 1) % self.cfg.opt_step_interval == 0:
            opt.step()  # type: ignore
            opt.zero_grad()  # type: ignore

        logits_cloned = logits.clone().detach()
        labels_cloned = labels.clone().detach()

        self._outputs["train"].append((logits_cloned, labels_cloned))
        self.log_on_step(logits_cloned, labels_cloned, tag="train")
        if (batch_idx + 1) % self.cfg.log_interval == 0:
            self.log_on_interval(tag="train")

    def training_epoch_end(self, outputs):
        self._epoch_end(tag="train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids, token_type_ids, labels = batch
            _, seq_len = input_ids.shape
            position_ids = torch.stack(
                [torch.arange(0, seq_len, dtype=torch.long, device=self.device)]
                * self.cfg.batchsize
            )
            logits = self(input_ids, token_type_ids, position_ids)

            self._outputs["val"].append((logits, labels))
            self.log_on_step(logits, labels, tag="val")
            if (batch_idx + 1) % self.cfg.log_interval == 0:
                self.log_on_interval(tag="train")
        return logits, labels

    def validation_epoch_end(self, outputs):
        self._epoch_end(tag="val")

    def log_on_interval(self, tag):
        logits = torch.cat([o[0] for o in self._outputs[tag]])
        labels = torch.cat([o[1] for o in self._outputs[tag]])
        self.vars[tag + "_metrics"].update(logits, labels.int())
        self.log(
            f"{tag}_loss",
            self.vars[tag + "_mean_loss"].compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
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

    def _epoch_end(self, tag="train"):

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
        self.vars[tag + "_metrics"].reset()
        self.vars[tag + "_confusion_matrix"].reset()
        self.vars[tag + "_roc"].reset()
        self.vars[tag + "_mean_loss"].reset()
        self._outputs[tag] = []
