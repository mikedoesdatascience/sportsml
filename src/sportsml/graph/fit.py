import lightning.pytorch as pl

from .datamodule import GraphDataModule
from .model import GraphModel


def fit(
    trainer: pl.Trainer, model: GraphModel, datamodule: GraphDataModule
):
    trainer.fit(model=model, datamodule=datamodule)
