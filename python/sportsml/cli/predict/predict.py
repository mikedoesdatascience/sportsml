import hydra
from omegaconf import DictConfig, OmegaConf

from ...utils.ensemble import ensemble_predict, graph_to_df


@hydra.main(version_base=None, config_path="conf", config_name="conf")
def predict(cfg: DictConfig) -> None:
    graph = hydra.utils.call(cfg.graph)
    team_map = hydra.utils.call(cfg.team_map)
    preds = ensemble_predict(graph, model_dir=cfg.model_dir)
    df = graph_to_df(preds, "neutral_pred", team_map)
    if cfg.sort:
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index, axis=1)
    df.to_csv(cfg.out)


if __name__ == "__main__":
    predict()
