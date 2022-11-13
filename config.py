cfg = {
    "num_encoder_blocks": 12,
    "num_attention_heads": 12,
    "d_model": 768,
    "vocab_size": 28996,
    "max_seq_len": 512,
    "p_dropout": 0.1,
    "batchsize": 32,
    "num_target_categories": 6,
}

label_tags = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
