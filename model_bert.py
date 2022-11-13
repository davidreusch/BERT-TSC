import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AdamW, get_scheduler


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
        # self.attention = BertSelfAttention(cfg)
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
        self.device = cfg.device

    # def forward(self, input_ids, token_type_ids):
    def forward(self, input_ids, token_type_ids):
        batchsize, seq_len = input_ids.shape
        position_ids = torch.stack(
            [torch.arange(0, seq_len, dtype=torch.long, device=self.device)] * batchsize
        )
        input_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        # print(f"{input_embeds.shape=}")
        # print(f"{token_type_embeds.shape=}")
        # print(f"{position_embeds.shape=}")
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

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor):
        embeds = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embeds)
        return encoder_output


class OutputLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.dense1 = nn.Linear(cfg.d_model, cfg.d_model // 2)
        self.act_fn = nn.GELU()
        self.dense2 = nn.Linear(cfg.d_model // 2, cfg.num_target_categories)
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(cfg.p_dropout)

    def forward(self, encoder_output: torch.Tensor):
        # encoder_output.shape: (batchsize, seq_len, d_model)
        x = self.dense1(
            encoder_output[:, 0]
        )  # use only hidden state corresponding to start of sequence token for classification
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.dense2(x)
        # x = self.softmax(x)
        return x


class ToxicSentimentClassificationModel(nn.Module):
    def __init__(self, cfg, state_dict) -> None:
        super().__init__()
        self.backbone = MyBertModel(cfg)
        self.backbone.load_state_dict(state_dict)
        self.output_layer = OutputLayer(cfg)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, **kwargs):
        backbone_output = self.backbone(input_ids, token_type_ids)
        return self.output_layer(backbone_output)


class TSCModel_PL(pl.LightningModule):
    def __init__(self, cfg, state_dict) -> None:
        super().__init__()
        self.backbone = MyBertModel(cfg)
        self.backbone.load_state_dict(state_dict)
        self.output_layer = OutputLayer(cfg)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, **kwargs):
        backbone_output = self.backbone(input_ids, token_type_ids)
        return self.output_layer(backbone_output)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.cfg.num_training_steps,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = self(input_ids[0], token_type_ids[0])
        loss = self.loss_fn(outputs, labels[0])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = self(input_ids[0], token_type_ids[0])
        loss = self.loss_fn(outputs, labels[0])
        predictions = (outputs > 0.5).type(torch.float)
        accuracy = torch.mean(torch.abs(predictions - labels))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        return loss
