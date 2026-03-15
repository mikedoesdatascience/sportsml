from typing import Any

import lightning.pytorch as pl
import mlflow.pyfunc
import pandas as pd
import scipy.stats
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
        ge = graph.edge_subgraph(graph.date < graph.date.max())
        gp = graph.edge_subgraph(graph.date == graph.date.max())

        x = self.encoder(edge_index=ge.edge_index, edge_attr=ge.edge_attr)
        preds = self.predictor(x, gp.edge_index)

        loss = torch.nn.functional.mse_loss(preds, gp.y)
        self.log(
            "train_loss", loss, prog_bar=True, on_epoch=True, batch_size=gp.num_edges
        )

        regression_metrics_output = self.train_regression_metrics(preds, gp.y)
        self.log_dict(regression_metrics_output, on_epoch=True, batch_size=gp.num_edges)

        classification_metrics_output = self.train_classification_metrics(
            preds > 0, gp.y > 0
        )
        self.log_dict(
            classification_metrics_output, on_epoch=True, batch_size=gp.num_edges
        )

        return loss

    def validation_step(self, graph, idx=None):
        ge = graph.edge_subgraph(graph.date < graph.date.max())
        gp = graph.edge_subgraph(graph.date == graph.date.max())

        x = self.encoder(edge_index=ge.edge_index, edge_attr=ge.edge_attr)
        preds = self.predictor(x, gp.edge_index)

        loss = torch.nn.functional.mse_loss(preds, gp.y)
        self.log(
            "val_loss", loss, prog_bar=True, on_epoch=True, batch_size=gp.num_edges
        )

        regression_metrics_output = self.val_regression_metrics(preds, gp.y)
        self.log_dict(
            regression_metrics_output,
            prog_bar=True,
            on_epoch=True,
            batch_size=gp.num_edges,
        )

        classification_metrics_output = self.val_classification_metrics(
            preds > 0, gp.y > 0
        )
        self.log_dict(
            classification_metrics_output,
            prog_bar=True,
            on_epoch=True,
            batch_size=gp.num_edges,
        )

    def test_step(self, graph, idx=None):
        ge = graph.edge_subgraph(graph.date < graph.date.max())
        gp = graph.edge_subgraph(graph.date == graph.date.max())

        x = self.encoder(edge_index=ge.edge_index, edge_attr=ge.edge_attr)
        preds = self.predictor(x, gp.edge_index)

        loss = torch.nn.functional.mse_loss(preds, gp.y)
        self.log(
            "test_loss", loss, prog_bar=True, on_epoch=True, batch_size=gp.num_edges
        )

        regression_metrics_output = self.test_regression_metrics(preds, gp.y)
        self.log_dict(
            regression_metrics_output,
            prog_bar=True,
            on_epoch=True,
            batch_size=gp.num_edges,
        )

        classification_metrics_output = self.test_classification_metrics(
            preds > 0, gp.y > 0
        )
        self.log_dict(
            classification_metrics_output,
            prog_bar=True,
            on_epoch=True,
            batch_size=gp.num_edges,
        )

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


class SportsMLPredictor(mlflow.pyfunc.PythonModel):
    def __init__(self, model: Any, team_embeddings: torch.Tensor):
        self.model = model
        self.team_embeddings = team_embeddings

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for team pairs.

        Args:
            context: MLFlow context (unused but required by PythonModel interface)
            model_input: DataFrame with columns [team_id, team_opp_id] or similar.
                        Column names should match what was used during training.
                        Can also include: season, date for filtering team stats.

        Returns:
            pd.DataFrame of predictions (one per row in model_input)
        """

        edge_index = torch.from_numpy(model_input[["opp", "team"]].T.values).long()

        preds = self.model.predictor(self.team_embeddings, edge_index=edge_index)
        result = model_input.assign(preds=preds.detach().numpy())
        result["prob"] = scipy.stats.norm.cdf(result["preds"] / result["preds"].std())
        return result
