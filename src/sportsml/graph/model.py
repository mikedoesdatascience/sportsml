import lightning.pytorch as pl
import torch
import torchmetrics

from .nn.encoder.edge_encoder import EdgeEncoder
from .nn.encoder.mean import EdgeMean
from .nn.predictor.ffn import EdgeFFN


class GraphModel(pl.LightningModule):
    def __init__(
        self,
        encoder: EdgeEncoder | EdgeMean,
        predictor: EdgeFFN,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

        self.lr = lr

        regression_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.MeanSquaredError(squared=False),
                torchmetrics.R2Score(),
                torchmetrics.PearsonCorrCoef(),
            ]
        )
        self.train_regression_metrics = regression_metrics.clone(prefix="train_")
        self.val_regression_metrics = regression_metrics.clone(prefix="val_")
        self.test_regression_metrics = regression_metrics.clone(prefix="test_")

        classification_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(task="binary"),
                torchmetrics.Precision(task="binary"),
                torchmetrics.Recall(task="binary"),
                torchmetrics.F1Score(task="binary"),
            ]
        )
        self.train_classification_metrics = classification_metrics.clone(
            prefix="train_"
        )
        self.val_classification_metrics = classification_metrics.clone(prefix="val_")
        self.test_classification_metrics = classification_metrics.clone(prefix="test_")

        self.save_hyperparameters(ignore=["encoder", "predictor"])
        self.hparams.update(
            {
                "encoder": encoder.hparams,
                "predictor": predictor.hparams,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, graph, idx=None):
        ge = graph.edge_subgraph(torch.where(graph.train_mask)[0])
        gp = graph.edge_subgraph(torch.where(~graph.train_mask)[0])

        x = self.encoder(edge_index=ge.edge_index, edge_attr=ge.edge_attr)
        preds = self.predictor(x, gp.edge_index)

        loss = torch.nn.functional.mse_loss(preds, gp.y)
        self.log("train_loss", loss, prog_bar=True)

        regression_metrics_output = self.train_regression_metrics(preds, gp.y)
        self.log_dict(regression_metrics_output, prog_bar=True)

        classification_metrics_output = self.train_classification_metrics(
            preds > 0, gp.y > 0
        )
        self.log_dict(classification_metrics_output, prog_bar=True)

        return loss

    def validation_step(self, graph, idx=None):
        ge = graph.edge_subgraph(torch.where(graph.train_mask)[0])
        gp = graph.edge_subgraph(torch.where(~graph.train_mask)[0])

        x = self.encoder(edge_index=ge.edge_index, edge_attr=ge.edge_attr)
        preds = self.predictor(x, gp.edge_index)

        loss = torch.nn.functional.mse_loss(preds, gp.y)
        self.log("val_loss", loss, prog_bar=True)

        regression_metrics_output = self.val_regression_metrics(preds, gp.y)
        self.log_dict(regression_metrics_output, prog_bar=True)

        classification_metrics_output = self.val_classification_metrics(
            preds > 0, gp.y > 0
        )
        self.log_dict(classification_metrics_output, prog_bar=True)

    def test_step(self, graph, idx=None):
        ge = graph.edge_subgraph(torch.where(graph.train_mask)[0])
        gp = graph.edge_subgraph(torch.where(~graph.train_mask)[0])

        x = self.encoder(edge_index=ge.edge_index, edge_attr=ge.edge_attr)
        preds = self.predictor(x, gp.edge_index)

        loss = torch.nn.functional.mse_loss(preds, gp.y)
        self.log("test_loss", loss, prog_bar=True)

        regression_metrics_output = self.test_regression_metrics(preds, gp.y)
        self.log_dict(regression_metrics_output, prog_bar=True)

        classification_metrics_output = self.test_classification_metrics(
            preds > 0, gp.y > 0
        )
        self.log_dict(classification_metrics_output, prog_bar=True)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        **kwargs,
    ):
        hparams = torch.load(checkpoint_path, weights_only=False)["hyper_parameters"]

        kwargs |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("encoder", "predictor")
        }

        return super().load_from_checkpoint(
            checkpoint_path, weights_only=False, **kwargs
        )
