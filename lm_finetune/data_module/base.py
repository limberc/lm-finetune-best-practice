import multiprocessing
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import (
    default_data_collator
)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super(BaseDataModule, self).__init__()
        self.batch_size = batch_size

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True,
                          collate_fn=default_data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True,
                          collate_fn=default_data_collator)
