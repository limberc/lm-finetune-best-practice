from os import path
from typing import Optional

import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Tokenizer
)

from .base import BaseDataModule


class NLPDataModule(BaseDataModule):
    def __init__(self,
                 batch_size: int,
                 data_name: str,
                 name_split: str,
                 save_tokenizer_path: str,
                 max_length: int = 1024):
        super(NLPDataModule, self).__init__(batch_size)
        self.data_name = data_name
        self.name_split = name_split
        self.save_tokenizer_path = save_tokenizer_path
        self.save_merge_path = save_tokenizer_path + '-merges.txt'
        self.save_vocab_path = save_tokenizer_path + '-vocab.json'
        self.max_length = max_length

    @staticmethod
    def batch_iterator(dataset, batch_size=1000):
        for key in dataset.keys():
            for i in range(0, len(dataset[key]), batch_size):
                yield dataset[key][i: i + batch_size]["text"]

    def tokenize_dataset(self):
        tokenizer = GPT2Tokenizer(vocab_file=self.save_vocab_path, merges_file=self.save_merge_path,
                                  unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<pad>')
        self.train_dataset = self.dataset['train'].map(
            lambda x: tokenizer(x['text'], padding='max_length', truncation=True,
                                max_length=self.max_length), batched=True)
        self.val_dataset = self.dataset['validation'].map(
            lambda x: tokenizer(x['text'], padding='max_length', truncation=True,
                                max_length=self.max_length), batched=True)
        cols = ['input_ids', 'attention_mask']
        self.train_dataset.set_format(type='torch', columns=cols)
        self.val_dataset.set_format(type='torch', columns=cols)
        torch.save(self.train_dataset, './data/{}_train.pt'.format(self.save_tokenizer_path))
        torch.save(self.val_dataset, './data/{}_val.pt'.format(self.save_tokenizer_path))

    def train_tokenizer(self):
        if not (path.exists(self.save_vocab_path) and path.exists(self.save_merge_path)):
            print("Training Tokenizer.")
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train_from_iterator(self.batch_iterator(self.dataset), min_frequency=2,
                                          special_tokens=[
                                              "<s>",
                                              "<pad>",
                                              "</s>",
                                              "<unk>",
                                              "<mask>",
                                          ])
            tokenizer.save_model(".", self.save_tokenizer_path)

    def prepare_data(self, *args, **kwargs):
        self.dataset = load_dataset(self.data_name, self.name_split)
        if "validation" not in self.dataset.keys():
            self.dataset['validation'] = load_dataset(
                self.data_name, self.name_split,
                split=f"train[:20%]"
            )
            self.dataset['train'] = load_dataset(
                self.data_name, self.name_split,
                split=f"train[20%:]"
            )
        self.train_tokenizer()

    def setup(self, stage: Optional[str] = None, data_dir="./data"):
        assert stage in ['train', 'fit']
        if stage == "train":
            self.prepare_data()
            self.tokenize_dataset()
        elif stage == 'fit':
            if not (path.exists("{}/{}_train.pt".format(data_dir, self.save_tokenizer_path)) and path.exists(
                    "{}/{}_val.pt".format(data_dir, self.save_tokenizer_path))):
                print("Dataset not founded.")
                print("Train the tokenizer from scratch.")
                return self.setup('train')
            self.train_dataset = torch.load("{}/{}_train.pt".format(data_dir, self.save_tokenizer_path))
            self.val_dataset = torch.load("{}/{}_val.pt".format(data_dir, self.save_tokenizer_path))


class OSCARDataModule(NLPDataModule):
    def __init__(self, batch_size):
        super(OSCARDataModule, self).__init__(batch_size, data_name='oscar', name_split='unshuffled_deduplicated_eo',
                                              save_tokenizer_path="esperberto")


class WikiText2DataModule(NLPDataModule):
    def __init__(self, batch_size):
        super(WikiText2DataModule, self).__init__(batch_size, data_name='wikitext', name_split='wikitext-2-v1',
                                                  save_tokenizer_path='wikitext2')
