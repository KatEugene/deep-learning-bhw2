from torch import nn
import torch
from src.utils.globals import (
    PAD_ID, SOS_ID, EOS_ID,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN,
)
from math import log


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_seq_len).reshape(-1, 1)
        dimension = torch.exp(-torch.arange(0, d_model, 2) * log(1e4) / d_model)

        pos_embedding = torch.zeros(size=(max_seq_len, d_model))
        pos_embedding[:, 0::2] = torch.sin(position * dimension)
        pos_embedding[:, 1::2] = torch.cos(position * dimension)
        self.register_buffer('pos_embedding', pos_embedding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        return self.dropout(batch + self.pos_embedding[:batch.size(1)])


class LanguageModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, decoders, max_seq_len, max_generated_length):
        super(LanguageModel, self).__init__()
        self.source_embedding = nn.Embedding(input_dim, d_model)
        self.target_embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers,
                                          num_decoder_layers, dim_feedforward, dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoders = decoders
        self.max_generated_length = max_generated_length

    def forward(self, src_text, trg_text):
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self._generate_masks(src_text, trg_text)

        src_embedding = self.source_embedding(src_text)
        trg_embedding = self.target_embedding(trg_text)

        src_pos_embedding = self.positional_encoding(src_embedding)
        trg_pos_embedding = self.positional_encoding(trg_embedding)

        output = self.transformer(src_pos_embedding, trg_pos_embedding,
                                  src_mask, trg_mask, None,
                                  src_padding_mask, trg_padding_mask, src_padding_mask)
        logits = self.fc_out(output)
        return {"logits": logits}

    @torch.inference_mode()
    def inference(self, **batch):
        device = next(self.parameters()).device
        texts = []
        for src in batch['src_text']:
            src = src.unsqueeze(0).to(device)
            trg = torch.tensor([[SOS_ID]]).to(device)
            for _ in range(self.max_generated_length):
                logits = self(src, trg)['logits'][0, -1]
                next_token = logits.argmax().item()
                trg = torch.cat([trg, torch.tensor([[next_token]]).to(device)], dim=1)
                if next_token == EOS_ID:
                    break
            texts.append(trg.squeeze().tolist())

        decoded_texts = {}
        decoded_texts["decoded_src_text"] = self._translate(batch["src_text"], "src")
        decoded_texts["decoded_trg_text"] = self._translate(batch["trg_text"], "trg")
        decoded_texts["decoded_translation_text"] = self._translate(texts, "trg")
        return decoded_texts

    def _translate(self, texts, lang):
        if len(texts[0]) == 0:
            return texts
        idx2text = self.decoders[lang].get_itos()
        decoded_texts = []
        for text in texts:
            decoded_text = list(map(lambda token: idx2text[token], text))
            decoded_text = ' '.join(decoded_text)
            for special_token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
                decoded_text = decoded_text.replace(special_token, '')
            decoded_texts.append(decoded_text)
        return decoded_texts

    def _generate_trg_mask(self, n):
        return torch.triu(torch.full(size=(n, n), fill_value=float('-inf')), diagonal=1)

    def _generate_masks(self, src, trg):
        device = next(self.parameters()).device

        src_mask = torch.zeros((src.size(1), src.size(1))).to(device)
        trg_mask = self._generate_trg_mask(trg.size(1)).to(device)
        src_padding_mask = src == PAD_ID
        trg_padding_mask = trg == PAD_ID

        return src_mask, trg_mask, src_padding_mask, trg_padding_mask
