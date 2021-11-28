import re
from typing import List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
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

        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.cls_token = cls_token
        self.cls_token_id = cls_token_id
        self.sep_token = sep_token
        self.sep_token_id = sep_token_id
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id

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
        desc: str = '',
    ) -> BatchEncoding:
        texts = [texts] if type(texts) == str else texts
        texts = [self.cls_token + text + self.sep_token for text in texts]

        encoded_text = []
        for text in tqdm(texts, desc=desc+'Encoding texts...'):
            encoded_text.append(self.encode(text))

        self.input_ids = pad_sequence(encoded_text, batch_first=True)
        self.token_type_ids = torch.zeros_like(
            self.input_ids, dtype=torch.int64)
        self.attention_mask = torch.where(
            self.input_ids != self.pad_token_id, 1, 0)

        return BatchEncoding({
            'input_ids': self.input_ids,
            'token_type_ids': self.token_type_ids,
            'attention_mask': self.attention_mask,
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
    def __init__(
        self,
        masked_texts: List[str],
        label_texts: List[str],
        data_verifying: bool = False,
        tokenizer=CharTokenizer(),
    ):
        if len(masked_texts) != len(label_texts):
            raise IndexError(
                f"'masked_texts' and 'label_texts' doesn't have same length. {len(masked_texts)} != {len(label_texts)}"
            )

        self.tokenizer = tokenizer

        batch_encoding = self.tokenizer(masked_texts, desc='Inputs: ')
        labels = self.tokenizer(label_texts, desc='Labels: ')['input_ids']

        if data_verifying:
            for m_ids, l_ids in zip(tqdm(batch_encoding['input_ids'], desc='Verifying data...'), labels):
                m_ids = [id for id in m_ids if id !=
                         self.tokenizer.pad_token_id]
                l_ids = [id for id in l_ids if id !=
                         self.tokenizer.pad_token_id]

                is_not_same_length = len(m_ids) != len(l_ids)

                for m_id, l_id in zip(m_ids, l_ids):
                    if is_not_same_length or (m_id != l_id and int(m_id) not in self.tokenizer.id2sptoken):
                        raise IndexError(
                            f"'masked_text' and 'label_text' aren't same. '{self.tokenizer.decode(m_ids)}' != '{self.tokenizer.decode(l_ids)}'"
                        )

        self.batch_encoding: BatchEncoding = batch_encoding
        self.batch_encoding['labels'] = labels

    def __len__(self):
        return len(list(self.batch_encoding.values())[0])

    def __getitem__(self, index):
        return{
            key: val[index] for key, val in self.batch_encoding.items()
        }


def mask_idx(text: str, idx: Union[int, List[int]]) -> str:
    text = list(text)
    if type(idx) == int:
        idx = [idx]
    for i in idx:
        text[i] = '[MASK]'
    return ''.join(text)


def mask_sents(sents_origin: List[str]):
    sents, sents_masked = [], []
    for sent in sents_origin:
        for i in range(len(sent)):
            if sent[i] == ' ':
                continue
            sents_masked.append(mask_idx(sent, i))
            sents.append(sent)
    return sents_masked, sents
