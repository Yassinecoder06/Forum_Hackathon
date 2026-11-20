import torch
from torch import nn

from .dataset import tokens

FRENCH_ENCODER = "cmarkea/distilcamembert-base"
FRENCH_REVISION = "82fb7671777bb680416031dcced25c7bfa6e859b"


def french_features(texts, device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(FRENCH_ENCODER, revision=FRENCH_REVISION)
    encoder = AutoModel.from_pretrained(FRENCH_ENCODER, revision=FRENCH_REVISION,
                                        use_safetensors=True).to(device).eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(texts), 8):
            inputs = tokenizer(texts[start:start+8], return_tensors="pt",
                               padding="max_length", truncation=True, max_length=48).to(device)
            hidden = encoder(**inputs).last_hidden_state
            result.append(hidden.masked_fill(inputs.attention_mask[...,None] == 0, 0))
    return torch.cat(result)


class FrozenFeatureProjection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(768, width)

    def forward(self, features):
        return self.projection(features), features.eq(0).all(dim=-1)


def encode(text, vocab, length=48):
    sequence = [vocab.get(token, 1) for token in tokens(text)[:length]] or [1]
    return sequence + [0] * (length - len(sequence))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, width):
        super().__init__()
        self.words = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position = nn.Parameter(torch.randn(48,width)*.02)
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*4,batch_first=True,
                                             norm_first=True),2,enable_nested_tensor=False)

    def forward(self, ids):
        padding = ids == 0
        memory = self.layers(self.words(ids)+self.position,src_key_padding_mask=padding)
        return memory, padding
