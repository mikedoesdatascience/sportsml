import itertools

import dgl
import lightning.pytorch as pl
import torch
import torchmetrics


class GraphModel(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        edge_encoder_features: str = "f",
        edge_targets: str = "y",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

        self.edge_encoder_features = edge_encoder_features
        self.edge_targets = edge_targets

        self.lr = lr

        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.mae = torchmetrics.MeanAbsoluteError()
        self.accuracy_score = torchmetrics.classification.BinaryAccuracy()

        self.save_hyperparameters(ignore=["encoder", "predictor"])
        self.hparams.update(
            {
                "encoder": encoder.hparams,
                "predictor": predictor.hparams,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1, total_iters=5
                ),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
            ],
            milestones=[2],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler},
        }

    def batch_step(self, g, p):
        g = g.local_var()
        p = p.local_var()

        h = self.encoder(g)
        p.ndata["h"] = h
        home_x, away_x = self.predictor(p)
        home_y = p.edges["home"].data["y"]
        away_y = p.edges["away"].data["y"]
        return torch.vstack([home_x, away_x]), torch.vstack([home_y, away_y])

    def training_step(self, graphs, idx):
        p, y = self.batch_step(*graphs)
        loss = torch.nn.functional.mse_loss(p, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, graphs, g_idx):
        p, y = self.batch_step(*graphs)
        self.rmse.update(p, y)
        self.mae.update(p, y)
        self.accuracy_score.update(p > 0, y > 0)

    def on_validation_epoch_end(self, *args, **kwargs):
        self.log("val_rmse", self.rmse.compute(), prog_bar=True)
        self.log("val_mae", self.mae.compute())
        self.log("val_accuracy", self.accuracy_score.compute(), prog_bar=True)
        self.rmse.reset()
        self.mae.reset()
        self.accuracy_score.reset()

    def test_step(self, graphs, g_idx):
        p, y = self.batch_step(*graphs)
        self.rmse.update(p, y)
        self.mae.update(p, y)
        self.accuracy_score.update(p > 0, y > 0)

    def on_test_epoch_end(self, *args, **kwargs):
        self.log("test_rmse", self.rmse.compute())
        self.log("test_mae", self.mae.compute())
        self.log("test_accuracy", self.accuracy_score.compute())
        self.rmse.reset()
        self.mae.reset()
        self.accuracy_score.reset()

    def predict(self, graph):
        h = self.encoder(graph)
        p = dgl.heterograph(
            {
                ("team", "home", "team"): list(
                    itertools.permutations(range(graph.number_of_nodes()), 2)
                ),
                ("team", "away", "team"): list(
                    itertools.permutations(range(graph.number_of_nodes()), 2)
                ),
            },
            {"team": graph.number_of_nodes()},
        )
        p.ndata["h"] = h
        home_p, away_p = self.predictor(p)
        p.edges["home"].data["pred"] = home_p
        p.edges["away"].data["pred"] = away_p
        return p

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        **kwargs,
    ):
        hparams = torch.load(checkpoint_path)["hyper_parameters"]

        kwargs |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("encoder", "predictor")
        }

        return super().load_from_checkpoint(checkpoint_path, **kwargs)
