import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        with open("config.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_ = str(self.texts[idx])

        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text_,
            add_special_tokens=self.config["preprocessing"]["add_special_tokens"],
            max_length=self.config["preprocessing"]["max_length"],
            return_token_type_ids=False,
            pad_to_max_length=self.config["preprocessing"]["pad_to_max_length"],
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return {
            "data": text_,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }
