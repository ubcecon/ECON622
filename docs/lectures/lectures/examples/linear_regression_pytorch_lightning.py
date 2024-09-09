# Example solving linear regression adding in:
# pytorch lightning wrapper, including CLI
# Can call on CLI with:
# python lectures/examples/linear_regression_pytorch_lightning.py --trainer.max_epochs=500 --optimizer.lr=0.0001 --model.N=500

import yaml
import math
import numpy as np
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

# Not required but removes annoying warning
import warnings
warnings.filterwarnings(
    "ignore",
    ".*does not have many workers which may be.*",
)

class LinearRegressionExample(pl.LightningModule):
    def __init__(
        self,
        N,
        M,
        sigma,
        train_prop,
        val_prop,
        batch_size,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(M, 1, bias=False)

    def residuals(self, X, Y):  # batches or full data
        Y_hat = self.model(X).squeeze()
        return ((Y_hat - Y) ** 2).mean()

    def training_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        loss = self.residuals(X_batch, Y_batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        loss = self.residuals(X_batch, Y_batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        loss = self.residuals(X_batch, Y_batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def setup(self, stage):
        M = self.hparams.M
        N = self.hparams.N
        if not stage == "test":
            self.theta = torch.randn(M)
            X = torch.randn(N, M)
            Y = X @ self.theta + self.hparams.sigma * torch.randn(N)  # Adding noise
            train_size = int(self.hparams.train_prop * N)
            val_size = int(self.hparams.val_prop * N)
            test_size = N - train_size - val_size
            self.train_data, self.val_data, self.test_data = random_split(
                TensorDataset(X, Y), [train_size, val_size, test_size]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.hparams.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.hparams.batch_size, shuffle=False
        )


if __name__ == "__main__":
    cli = LightningCLI(
        LinearRegressionExample,
        seed_everything_default=123,
        run=False,
        save_config_callback=None,
        parser_kwargs={
            "default_config_files": ["lectures/examples/linear_regression_pytorch_lightning_defaults.yaml"]
        },
        save_config_kwargs={"save_config_overwrite": True},
    )
    # After fitting the model, check the test suite
    cli.trainer.fit(cli.model)
    cli.trainer.test(cli.model)
