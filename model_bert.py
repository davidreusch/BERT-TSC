import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelROC,
    MultilabelAUROC,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelConfusionMatrix,
)
from torchmetrics import MeanMetric, MetricCollection
from torch.optim import AdamW
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
        querys_proj = self.query(seq).view(
            batchsize, seq_len, self.num_attention_heads, d_v
        )
        querys_proj = querys_proj.transpose(1, 2)
        keys_proj = self.key(seq).view(
            batchsize, seq_len, self.num_attention_heads, d_v
        )
        keys_proj = keys_proj.transpose(1, 2)
        values_proj = self.value(seq).view(
            batchsize, seq_len, self.num_attention_heads, d_v
        )
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
        self.LayerNorm = nn.LayerNorm(
            (cfg.d_model,), eps=1e-12, elementwise_affine=True
        )
        self.dropout = nn.Dropout(cfg.p_dropout)

    def forward(self, attention_output):
        # attention_output.shape = (batchsize, seq_len, d_model)
        # in which order to apply these things? see below
        return (
            self.dropout(self.LayerNorm(self.dense(attention_output)))
            + attention_output
        )


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
        self.LayerNorm = nn.LayerNorm(
            (cfg.d_model,), eps=1e-12, elementwise_affine=True
        )
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
        self.LayerNorm = nn.LayerNorm(
            (cfg.d_model,), eps=1e-12, elementwise_affine=True
        )
        self.dropout = nn.Dropout(cfg.p_dropout)
        self.register_buffer(
            "position_ids", torch.arange(cfg.max_seq_len).expand((1, -1))
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        batchsize, seq_len = input_ids.shape
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
    def __init__(self, cfg, state_dict) -> None:
        super().__init__()
        self.backbone = MyBertModel(cfg)
        self.backbone.load_state_dict(state_dict)
        self.output_layer = OutputLayer(cfg)
        # ratio of positive to negative samples is 1/27, so add this as loss weights (should be adapted for each class individually)
        positive_weights = 27 * torch.ones(cfg.num_target_categories)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
        self.cfg = cfg
        self.automatic_optimization = False
        self._train_outputs = []
        self._val_outputs = []

        metrics = MetricCollection(
            [
                MultilabelAccuracy(num_labels=6, average=None),
                MultilabelAUROC(num_labels=6, average=None),
                MultilabelPrecision(num_labels=6, average=None),
                MultilabelRecall(num_labels=6, average=None),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.train_roc = MultilabelROC(num_labels=6)
        self.val_roc = MultilabelROC(num_labels=6)
        self.train_confusion_matrix = MultilabelConfusionMatrix(num_labels=6)
        self.val_confusion_matrix = MultilabelConfusionMatrix(num_labels=6)

        self.mean_train_loss = MeanMetric()
        self.mean_val_loss = MeanMetric()

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
        seq_len = input_ids.shape[2]
        batchsize = input_ids.shape[1]
        position_ids = torch.stack(
            [torch.arange(0, seq_len, dtype=torch.long, device=self.device)] * batchsize
        )
        logits = self(input_ids[0], token_type_ids[0], position_ids=position_ids)
        loss_val = self.loss_fn(logits, labels[0])
        loss_val.backward()

        # accumulate gradients of multiple batches
        if (batch_idx + 1) % self.cfg.opt_step_interval == 0:
            opt.step()
            opt.zero_grad()

        self.mean_train_loss.update(loss_val.item())
        self.train_metrics.update(logits, labels[0].int())
        self.train_confusion_matrix.update(logits, labels[0].int())
        self.log(
            "train_loss",
            self.mean_train_loss.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_accuracy",
            self.train_metrics.compute()["train_MultilabelAccuracy"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self._train_outputs.append((logits, labels[0].int()))

    def training_epoch_end(self, outputs):

        train_dict = self.train_metrics.compute()
        auroc = train_dict["train_MultilabelAUROC"]
        cm = self.train_confusion_matrix.compute()

        logits = [o[0] for o in self._train_outputs]
        labels = [o[1] for o in self._train_outputs]
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        fpr, tpr, thresholds = self.train_roc(logits, labels)

        tensorboard = self.logger.experiment
        utils.log_roc_curve(
            fpr,
            tpr,
            tensorboard,
            self.current_epoch,
            tag="train",
            auroc=auroc,
        )
        utils.log_confusion_matrix(cm, tensorboard, self.current_epoch, tag="train")
        utils.log_metrics_table(
            train_dict, tensorboard, self.current_epoch, tag="train"
        )
        self.train_metrics.reset()
        self.mean_train_loss.reset()
        self.train_roc.reset()
        self.train_confusion_matrix.reset()
        self._train_outputs = []

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, labels = batch
        seq_len = input_ids.shape[2]
        position_ids = torch.stack(
            [torch.arange(0, seq_len, dtype=torch.long, device=self.device)]
            * self.cfg.batchsize
        )
        logits = self(input_ids[0], token_type_ids[0], position_ids)
        loss_val = self.loss_fn(logits, labels[0])

        self.mean_val_loss.update(loss_val.item())
        self.val_metrics.update(logits, labels[0].int())
        self.val_confusion_matrix.update(logits, labels[0].int())
        self.log(
            "val_loss",
            self.mean_val_loss.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_accuracy",
            self.val_metrics.compute()["val_MultilabelAccuracy"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if (batch_idx + 1) % 5 == 0:
            pass

        self._val_outputs.append((logits, labels[0].int()))
        return logits, labels[0].int()

    def validation_epoch_end(self, outputs):

        val_dict = self.val_metrics.compute()
        auroc = val_dict["val_MultilabelAUROC"]
        cm = self.val_confusion_matrix.compute()

        logits = [o[0] for o in self._val_outputs]
        labels = [o[1] for o in self._val_outputs]
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        fpr, tpr, thresholds = self.val_roc(logits, labels)
        tensorboard = self.logger.experiment
        utils.log_roc_curve(
            fpr,
            tpr,
            tensorboard,
            self.current_epoch,
            tag="val",
            auroc=auroc,
        )
        utils.log_confusion_matrix(cm, tensorboard, self.current_epoch, tag="val")
        utils.log_metrics_table(val_dict, tensorboard, self.current_epoch, tag="val")
        self.val_metrics.reset()
        self.mean_val_loss.reset()
        self.val_roc.reset()
        self.val_confusion_matrix.reset()
        self._val_outputs = []
