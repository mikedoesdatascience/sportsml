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
        edge_predictor_features: str = "p",
        edge_targets: str = "y",
        train_mask: str = "train",
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

        self.edge_encoder_features = edge_encoder_features
        self.edge_predictor_features = edge_predictor_features
        self.edge_targets = edge_targets
        self.train_mask = train_mask

        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.mae = torchmetrics.MeanAbsoluteError()
        self.accuracy_score = torchmetrics.classification.MulticlassAccuracy(
            num_classes=2
        )
        self.precision_score = torchmetrics.classification.MulticlassPrecision(
            num_classes=2
        )
        self.recall_score = torchmetrics.classification.MulticlassRecall(num_classes=2)

        self.save_hyperparameters(
            "encoder",
            "predictor",
            "edge_encoder_features",
            "edge_predictor_features",
            "edge_targets",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1, total_iters=2
                ),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7),
            ],
            milestones=[2],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler}}

    def batch_step(self, g):
        train = g.edge_subgraph(
            g.edata[self.train_mask] == True, relabel_nodes=False
        ).local_var()
        test = g.edge_subgraph(
            g.edata[self.train_mask] == False, relabel_nodes=False
        ).local_var()
        h = self.encoder(train, train.edata[self.edge_encoder_features])
        e = test.edata[self.edge_predictor_features]
        p = self.predictor(test, h, e)
        y = test.edata[self.edge_targets]
        return p, y

    def training_step(self, g, g_idx):
        p, y = self.batch_step(g)
        loss = torch.nn.functional.mse_loss(p, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, g, g_idx):
        p, y = self.batch_step(g)
        self.rmse(p, y)
        self.mae(p, y)
        self.accuracy_score(p > 0, y > 0)
        self.precision_score(p > 0, y > 0)
        self.recall_score(p > 0, y > 0)
        self.log("val_rmse", self.rmse, prog_bar=True)
        self.log("val_mae", self.mae)
        self.log("val_accuracy", self.accuracy_score)
        self.log("val_precision", self.precision_score)
        self.log("val_recall", self.recall_score)

    def test_step(self, g, g_idx):
        p, y = self.batch_step(g)
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

    def predict(self, graph):
        h = self.encoder(graph, graph.edata[self.edge_encoder_features])
        pred_graph = dgl.graph(
            list(itertools.permutations(range(graph.number_of_nodes()), 2))
        )
        pred_graph.edata["home_pred"] = self.predictor(
            pred_graph, h, torch.ones((pred_graph.number_of_edges()), 1)
        )
        pred_graph.edata["away_pred"] = self.predictor(
            pred_graph, h, torch.zeros((pred_graph.number_of_edges()), 1)
        )
        return pred_graph
