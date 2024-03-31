import datetime

import hydra
import numpy as np
from omegaconf import DictConfig

from ...mongo import client
from ...mongo.model_store import load_graph_model
from ...utils.ensemble import load_models, predict_with_model


@hydra.main(version_base=None, config_path="conf", config_name="conf")
def predict(cfg: DictConfig) -> None:
    graph = hydra.utils.call(cfg.graph)
    team_map = hydra.utils.call(cfg.team_map)
    if cfg.model == "latest":
        model = load_graph_model(
            client[cfg.sport].models.find({}).sort("date", -1).limit(1).next()
        )
        models = [model]
    else:
        models = load_models(cfg.model)
    _ = [model.eval() for model in models]
    preds = [predict_with_model(model, graph, team_map, sort=False) for model in models]
    avg_preds = np.stack([p.values for p in preds]).mean(axis=0)
    df = preds[0].copy()
    df[:] = avg_preds
    if cfg.sort:
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index, axis=1)
    if cfg.db:
        client[cfg.sport].predictions.update_one(
            {"_id": datetime.date.today().isoformat()},
            {"$set": {"predictions": df.to_dict()}},
            upsert=True,
        )
    if cfg.out:
        df.to_csv(cfg.out)


if __name__ == "__main__":
    predict()
