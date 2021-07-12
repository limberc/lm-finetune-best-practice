from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from lm_finetune import FineTuneModel, NLPDataModule


def run_cli():
    parent_parser = ArgumentParser()
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--seed', type=int, default=42)
    parent_parser = NLPDataModule.add_argparse_args(parent_parser)
    parser = FineTuneModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler="simple",
        precision=16,
        accelerator='ddp',
        deterministic=True,
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    args = parser.parse_args()
    main(args)


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)
    dm = NLPDataModule.from_argparse_args(args)
    model = FineTuneModel(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)


if __name__ == '__main__':
    run_cli()
