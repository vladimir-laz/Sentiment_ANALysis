import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
from transformers import (
    BertTokenizerFast,
    RobertaTokenizer,
    GPT2Tokenizer,
    DistilBertTokenizer,
    RobertaTokenizerFast
)
from dataset import TextClassificationDataset
import torch
from torch.utils.data import Dataset, DataLoader


class Preprocessing:
    def __init__(self):
        """
        Initializes the Preprocessing class. It reads the configuration from "config.yaml",
        loads the data based on the dataset name specified in the configuration,
        generates category-integer mapping if required, splits the data if required,
        and tokenizes the data using the specified tokenizer.
        """
        with open("config.yaml") as f:
            self.full_config = yaml.load(f, Loader=yaml.FullLoader)
            self.config = self.full_config["preprocessing"]

        if self.full_config["general"]["dataset_name"] == "bbc":
            self.df = pd.read_csv("data/bbc-text.csv")
        elif self.full_config["general"]["dataset_name"] == "ruSentNE":
            self.df = pd.read_csv("data/data_ruSentNE.csv")
        elif self.full_config["general"]["dataset_name"] == "ruSentNE_lemmatized":
            self.df = pd.read_csv("data/data_ruSentNE_lemmatized.csv")
        else:
            raise ValueError(
                f"Dataset {self.full_config['general']['dataset_name']} not supported"
            )

        if self.config["generate_mapping"]:
            self.get_mapping()

        if self.config["split"]:
            self.split_data()

        self.tokenize()

    def get_mapping(self):
        """
        Generates a dictionary that maps unique category labels in the dataset to integer values.
        If the "save_mapping" option is set to True in the configuration, it saves the mapping to "mapping.yaml".
        """
        # keys = self.df["category"].unique()
        mapping = {
            -1: 0,
            0: 1,
            1: 2
        }
        # mapping = {key: i for i, key in enumerate(keys)}
        self.df["label"] = self.df["category"].map(mapping)

        if self.config["save_mapping"]:
            with open(f"mapping.yaml", "w") as f:
                yaml.dump(mapping, f)

    def split_data(self):
        """
        Splits the data into a training set and a validation set.
        The split ratio, seed for random shuffling, and whether to perform stratified sampling are
        all determined by the configuration. Stratified sampling is used if "split_stratify" is set to True.
        """
        if self.config["split_stratify"]:
            self.train_df, self.val_df = train_test_split(
                self.df,
                test_size=1 - self.config["split_ratio"],
                random_state=self.config["split_seed"],
                shuffle=self.config["split_shuffle"],
                stratify=self.df["label"],
            )
        else:
            self.train_df, self.val_df = train_test_split(
                self.df,
                test_size=1 - self.config["split_ratio"],
                random_state=self.config["split_seed"],
                shuffle=self.config["split_shuffle"],
            )

    def tokenize(self):
        """
        Tokenizes the text data in the DataFrame using the tokenizer specified in the configuration.
        Raises a ValueError if the specified tokenizer is not supported.
        Currently supports "BertTokenizerFast", "RobertaTokenizer", "GPT2Tokenizer", and "DistilBertTokenizer".
        """
        if self.config["tokenizer"] == "BertTokenizerFast":
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        elif self.config["tokenizer"] == "RobertaTokenizerFast":
            self.tokenizer = RobertaTokenizerFast.from_pretrained('blinoff/roberta-base-russian-v0', max_len=512)
        elif self.config["tokenizer"] == "RobertaTokenizer":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        elif self.config["tokenizer"] == "GPT2Tokenizer":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        elif self.config["tokenizer"] == "DistilBertTokenizer":
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
        else:
            raise ValueError(f"Tokenizer {self.config['tokenizer']} not supported")

    def get_datasets(self):
        """Creates TextClassificationDataset objects for the training and validation data."""
        train_dataset = TextClassificationDataset(
            texts=self.train_df["text"].to_numpy(),
            labels=self.train_df["label"].to_numpy(),
            tokenizer=self.tokenizer,
        )

        val_dataset = TextClassificationDataset(
            texts=self.val_df["text"].to_numpy(),
            labels=self.val_df["label"].to_numpy(),
            tokenizer=self.tokenizer,
        )

        return {"train_set": train_dataset, "val_set": val_dataset}

    def get_dataloaders(self):
        """
        Creates PyTorch DataLoader objects for the training and validation data.
        The batch size and number of workers are determined by the configuration.
        The data is shuffled for the training DataLoader if "dataloader_shuffle" is set to True.
        Returns a dictionary containing the training and validation DataLoaders.
        """
        train_dataset = TextClassificationDataset(
            texts=self.train_df["text"].to_numpy(),
            labels=self.train_df["label"].to_numpy(),
            tokenizer=self.tokenizer,
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.full_config["general"]["batch_size"],
            shuffle=self.full_config["general"]["dataloader_shuffle"],
            num_workers=self.full_config["general"]["num_workers"],
        )
        val_dataset = TextClassificationDataset(
            texts=self.val_df["text"].to_numpy(),
            labels=self.val_df["label"].to_numpy(),
            tokenizer=self.tokenizer,
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=self.full_config["general"]["batch_size"],
            shuffle=False,
            num_workers=self.full_config["general"]["num_workers"],
        )

        return {"train_dl": train_dl, "val_dl": val_dl}
