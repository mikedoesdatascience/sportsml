import lightning.pytorch as pl
import mlflow.pyfunc

from .datamodule import GraphDataModule
from .model import GraphModel, SportsMLPredictor


def fit(
    trainer: pl.Trainer, model: GraphModel, datamodule: GraphDataModule, save_dir: str
):
    trainer.fit(model=model, datamodule=datamodule)

    if datamodule.test_ds:
        trainer.test(
            model=model, datamodule=datamodule, ckpt_path="best", weights_only=False
        )

    model = GraphModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    ge = datamodule.get_latest_graph()

    team_embeddings = model.encoder(edge_attr=ge.edge_attr, edge_index=ge.edge_index)

    predictor = SportsMLPredictor(model=model, team_embeddings=team_embeddings)

    mlflow.pyfunc.save_model(save_dir, python_model=predictor)
