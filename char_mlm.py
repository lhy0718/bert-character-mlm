import re
from typing import Dict, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import BatchEncoding


class CharTokenizer():
    def __init__(
        self,
        pad_token: str = '[PAD]',
        pad_token_id: str = 0,
        cls_token: str = '[CLS]',
        cls_token_id: str = 101,
        sep_token: str = '[SEP]',
        sep_token_id: str = 102,
        mask_token: str = '[MASK]',
        mask_token_id: str = 103,
    ):
        self.SPECIAL_TOKENS_ATTRIBUTES = [
            'pad_token',
            'cls_token',
            'sep_token',
            'mask_token',
        ]
        for token_name in self.SPECIAL_TOKENS_ATTRIBUTES:
            setattr(self, token_name, locals()[token_name])
            setattr(self, token_name + '_id', locals()[token_name + '_id'])
        self.special_tokens_map = {
            token_name: getattr(self, token_name)
            for token_name in self.SPECIAL_TOKENS_ATTRIBUTES
        }
        self.id2sptoken = {
            getattr(self, token_name + '_id'): getattr(self, token_name)
            for token_name in self.SPECIAL_TOKENS_ATTRIBUTES
        }
        self.sptoken2id = {
            getattr(self, token_name): getattr(self, token_name + '_id')
            for token_name in self.SPECIAL_TOKENS_ATTRIBUTES
        }

        self.char_id_starts = 200

    def __call__(
        self, texts: Union[List[str], str],
    ) -> BatchEncoding:
        texts = [texts] if type(texts) == str else texts

        encoded_text = [
            self.encode(text) for text in texts
        ]

        self.input_ids = pad_sequence(encoded_text, batch_first=True)
        self.token_type_ids = torch.zeros_like(
            self.input_ids, dtype=torch.int64)
        self.attention_mask = torch.zeros_like(
            self.input_ids, dtype=torch.int64)

        for i, ids in enumerate(self.input_ids):
            for j, id in enumerate(ids):
                if id > 0:
                    self.attention_mask[i][j] = 1

        return BatchEncoding({
            'input_ids': self.input_ids,
            'token_type_ids': self.token_type_ids,
            'attention_mask': self.attention_mask
        })

    def encode(
        self,
        text: str,
    ) -> torch.Tensor:
        sptokens_indice = []
        regex = '|'.join(self.id2sptoken.values()).replace('[', '\[')
        s = re.search(regex, text)
        while s:
            sptokens_indice.append((s.group(), s.start()))
            text = re.sub(regex, '_', text, 1)
            s = re.search(regex, text)

        text = list(text)
        for sptoken, i in sptokens_indice:
            text[i] = self.sptoken2id[sptoken]
        ids = torch.tensor([
            ord(char) + self.char_id_starts
            if type(char) == str else char for char in text
        ], dtype=torch.int64)
        return ids

    def batch_decode(
        self,
        input_ids: Union[List[int], List[List[int]],
                         "np.ndarray", "torch.Tensor", "tf.Tensor"]
    ) -> List[str]:
        return [
            self.decode(seq)
            for seq in input_ids
        ]

    def decode(
        self,
        seq: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
    ) -> str:
        return ''.join([
            chr(int(id) - self.char_id_starts)
            if id >= self.char_id_starts else self.id2sptoken[int(id)]
            for id in seq
        ])


class CharMLMDataset(Dataset):
    def __init__(self, masked_texts: List[str], label_texts: List[str]):
        self.tokenizer = CharTokenizer()
        self.batch_encoding =

    def __len__(self):
        pass

    def __getitem__(self, index) -> torch.Tensor:
        pass
