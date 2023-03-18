import dgl
import pytorch_lightning as pl
import torch
import torchmetrics


class GraphModel(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        edge_encoder_features: str = 'f',
        edge_predictor_features: str = 'p',
        edge_targets: str = 'y',
        train_mask: str = 'train'
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
        self.accuracy_score = torchmetrics.classification.MulticlassAccuracy(num_classes=2)
        self.precision_score = torchmetrics.classification.MulticlassPrecision(num_classes=2)
        self.recall_score = torchmetrics.classification.MulticlassRecall(num_classes=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def batch_step(self, g):
        train = g.edge_subgraph(g.edata[self.train_mask] == True, relabel_nodes=False).local_var()
        test = g.edge_subgraph(g.edata[self.train_mask] == False, relabel_nodes=False).local_var()
        h = self.encoder(train, train.edata[self.edge_encoder_features])
        e = test.edata[self.edge_predictor_features]
        p = self.predictor(test, h, e)
        y = test.edata[self.edge_targets]
        return p, y
    
    def training_step(self, g, g_idx):
        p, y = self.batch_step(g)
        loss = torch.nn.functional.mse_loss(p, y)
        return loss

    def validation_step(self, g, g_idx):
        p, y = self.batch_step(g)
        self.rmse(p, y)
        self.mae(p, y)
        self.accuracy_score(p > 0, y > 0)
        self.precision_score(p > 0, y > 0)
        self.recall_score(p > 0, y > 0)
        self.log('val_rmse', self.rmse)
        self.log('val_mae', self.mae)
        self.log('val_accuracy', self.accuracy_score)
        self.log('val_precision', self.precision_score)
        self.log('val_recall', self.recall_score)

    def test_step(self, g, g_idx):
        p, y = self.batch_step(g)
        self.rmse(p, y)
        self.mae(p, y)
        self.accuracy_score(p > 0, y > 0)
        self.precision_score(p > 0, y > 0)
        self.recall_score(p > 0, y > 0)
        self.log('test_rmse', self.rmse)
        self.log('test_mae', self.mae)
        self.log('test_accuracy', self.accuracy_score)
        self.log('test_precision', self.precision_score)
        self.log('test_recall', self.recall_score)
