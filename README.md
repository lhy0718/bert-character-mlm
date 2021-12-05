# bert-character-mlm

character tokenizer using BertTokenizer (uncased)

![bert-char-mlm](https://user-images.githubusercontent.com/11364584/142732816-81e1fbe8-c665-4351-bab3-cd00b1659b61.png)

## Usages

### Charcter tokenizer & Character MLM

```python
from transformers import AutoTokenizer, BertForMaskedLM, BertConfig

MODEL_NAME = 'char-bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

config = BertConfig(vocab_size=len(tokenizer))
model = BertForMaskedLM(config)
```
