import torch
import torchmetrics
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, in_feats, dims=[100, 100], out_feats=1, dropout=0.1):
        super().__init__()
        self.dims = list(dims)
        self.dims.insert(0, in_feats)
        layers = [torch.nn.BatchNorm1d(in_feats)]
        for idx in range(len(self.dims) - 1):
            layers.append(torch.nn.Linear(self.dims[idx], self.dims[idx + 1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.BatchNorm1d(self.dims[idx + 1]))
        layers.append(torch.nn.Linear(self.dims[-1], out_feats))
        self.mlp = torch.nn.Sequential(*layers)

        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.mae = torchmetrics.MeanAbsoluteError()
        self.accuracy_score = torchmetrics.classification.MulticlassAccuracy(
            num_classes=2
        )
        self.precision_score = torchmetrics.classification.MulticlassPrecision(
            num_classes=2
        )
        self.recall_score = torchmetrics.classification.MulticlassRecall(num_classes=2)

        self.save_hyperparameters("in_feats", "dims")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        loss = torch.nn.functional.mse_loss(p, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        self.rmse(p, y)
        self.mae(p, y)
        self.accuracy_score(p > 0, y > 0)
        self.precision_score(p > 0, y > 0)
        self.recall_score(p > 0, y > 0)
        self.log("val_rmse", self.rmse)
        self.log("val_mae", self.mae)
        self.log("val_accuracy", self.accuracy_score)
        self.log("val_precision", self.precision_score)
        self.log("val_recall", self.recall_score)

    def test_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        self.rmse(p, y)
        self.mae(p, y)
        self.accuracy_score(p > 0, y > 0)
        self.precision_score(p > 0, y > 0)
        self.recall_score(p > 0, y > 0)
        self.log("test_rmse", self.rmse)
        self.log("test_mae", self.mae)
        self.log("test_accuracy", self.accuracy_score)
        self.log("test_precision", self.precision_score)
        self.log("test_recall", self.recall_score)
