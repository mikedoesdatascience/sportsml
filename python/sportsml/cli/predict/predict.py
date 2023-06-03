import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from ...mongo import client
from ...mongo.model_store import load_graph_model
from ...utils.ensemble import ensemble_predict, graph_to_df


@hydra.main(version_base=None, config_path="conf", config_name="conf")
def predict(cfg: DictConfig) -> None:
    graph = hydra.utils.call(cfg.graph)
    team_map = hydra.utils.call(cfg.team_map)
    if cfg.model == "latest":
        model = load_graph_model(
            client[cfg.sport].models.find({}).sort("date", -1).limit(1).next()
        )
        preds = ensemble_predict(graph, models=[model])
    else:
        preds = ensemble_predict(graph, model_dir=cfg.model)
    df = graph_to_df(preds, "neutral_pred", team_map)
    if cfg.sort:
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index, axis=1)
    client[cfg.sport].predictions.update_one(
        {"_id": datetime.date.today().isoformat()},
        {"$set": {"predictions": df.to_dict()}},
        upsert=True,
    )


if __name__ == "__main__":
    predict()
