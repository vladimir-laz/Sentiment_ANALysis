import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        """
        Initializes the TextClassificationDataset class.

        Args:
            texts (list): A list of texts to be used for training.
            labels (list): A list of labels corresponding to the texts.
            tokenizer (transformers.PreTrainedTokenizer): A tokenizer for encoding the texts.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        with open("config.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the original text, input ids, attention mask, and label.
        """
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
