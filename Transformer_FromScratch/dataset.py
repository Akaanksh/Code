import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer_from, tokenizer_to, lang_from, lang_to, max_len):
        super().__init__()

        self.data = data
        self.tokenizer_from = tokenizer_from
        self.tokenizer_to = tokenizer_to
        self.lang_from = lang_from
        self.lang_to = lang_to
        self.max_len = max_len

        self.token_sos = torch.tensor([tokenizer_to.token_to_id("[SOS]")], dtype=torch.int64)
        self.token_eos = torch.tensor([tokenizer_to.token_to_id("[EOS]")], dtype=torch.int64)
        self.token_pad = torch.tensor([tokenizer_to.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text_from = item['translation'][self.lang_from]
        text_to = item['translation'][self.lang_to]

        tokens_from = self.tokenizer_from.encode(text_from).ids
        tokens_to = self.tokenizer_to.encode(text_to).ids

        pad_len_from = self.max_len - len(tokens_from) - 2
        pad_len_to = self.max_len - len(tokens_to) - 1

        if pad_len_from < 0 or pad_len_to < 0:
            raise ValueError("Input sequence too long")

        input_encoder = torch.cat([
            self.token_sos,
            torch.tensor(tokens_from, dtype=torch.int64),
            self.token_eos,
            torch.tensor([self.token_pad] * pad_len_from, dtype=torch.int64)
        ], dim=0)

        input_decoder = torch.cat([
            self.token_sos,
            torch.tensor(tokens_to, dtype=torch.int64),
            torch.tensor([self.token_pad] * pad_len_to, dtype=torch.int64)
        ], dim=0)

        label_sequence = torch.cat([
            torch.tensor(tokens_to, dtype=torch.int64),
            self.token_eos,
            torch.tensor([self.token_pad] * pad_len_to, dtype=torch.int64)
        ], dim=0)

        assert input_encoder.size(0) == self.max_len
        assert input_decoder.size(0) == self.max_len
        assert label_sequence.size(0) == self.max_len

        encoder_mask = (input_encoder != self.token_pad).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (input_decoder != self.token_pad).unsqueeze(0).int() & build_causal_mask(input_decoder.size(0))

        return {
            "encoder_input": input_encoder,
            "decoder_input": input_decoder,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label_sequence,
            "src_text": text_from,
            "tgt_text": text_to,
        }


def build_causal_mask(seq_len):
    mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).to(dtype=torch.int)
    return mask == 0
