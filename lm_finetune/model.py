from argparse import ArgumentParser

import pytorch_lightning as pl
from transformers import AdamW, AutoModelForCausalLM

from .stlr import STLR


class FineTuneModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = AutoModelForCausalLM.from_pretrained()
        self.model.lm_loss.reset_parameters()

    def configure_optimizers(self):
        model = self.model
        # According to https://arxiv.org/abs/1801.06146
        # tune each layer with different learning rates.
        # with l-th layer.
        # We have got \eta^{l-1}=\eta^{l}/2.6
        def _get_lr(n_layer):
            if not n_layer:
                return self.hparams.max_lr
            else:
                return _get_lr(n_layer - 1) / 2.6

        lrs = [_get_lr(layer) for layer in range(self.model.config.n_layer)]
        layer_ns = ['transformer.layer.{}.'.format(i) for i in range(self.model.config.n_layer)][::-1]
        residual_ns = [n for n, p in model.named_parameters() if 'transformer.layer.' not in n]
        optimizer_grouped_parameters = []
        for lr, layer_n in zip(lrs, layer_ns):
            optimizer_grouped_parameters.append(
                {
                    'params': [p for n, p in model.named_parameters() if layer_n in n],
                    'lr': lr,
                }
            )
        optimizer_grouped_parameters.append(
            {
                'params': [p for n, p in model.named_parameters() if any(
                    nt in n for nt in residual_ns)]
            }
        )

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.max_lr,
                          eps=self.hparams.adam_epsilon)
        scheduler = {
            'scheduler': STLR(optimizer, self.hparams.max_lr, self.total_steps),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--max_lr", default=2e-5, type=float,
                            help="max lr for STLR.", )
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser
