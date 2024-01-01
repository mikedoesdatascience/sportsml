import pathlib

import dgl
import pandas as pd
import torch

from ..models.gnn import GraphModel


def load_models(model_dir, model_class=GraphModel):
    ckpts = pathlib.Path(model_dir).rglob("*.ckpt")
    return [model_class.load_from_checkpoint(ckpt) for ckpt in ckpts]


def ensemble_predict(graph, models=None, model_dir=None, model_class=GraphModel):
    if not models and model_dir is not None:
        models = load_models(model_dir=model_dir, model_class=model_class)

    if not models:
        raise ValueError("cannot load models")

    preds = [model.predict(graph) for model in models]

    pred_graph = dgl.graph(preds[0].edges())

    pred_graph.edata["home_pred"] = torch.stack(
        [p.edata["home_pred"] for p in preds]
    ).mean(axis=0)
    pred_graph.edata["away_pred"] = torch.stack(
        [p.edata["away_pred"] for p in preds]
    ).mean(axis=0)

    pred_graph.edata["neutral_pred"] = (
        pred_graph.edata["home_pred"] + pred_graph.edata["away_pred"]
    ) / 2

    return pred_graph


def graph_to_df(graph, f, name_map=None):
    df = pd.DataFrame()
    u, v = graph.edges()
    df["team"] = v
    df["opp"] = u
    df[f] = graph.edata[f].detach()
    df = df.pivot_table(index="team", columns="opp", values=f)
    if name_map is not None:
        df = df.rename(index=name_map, columns=name_map)
    return df


def predict_with_model(model, graph, name_map=None, sort=True):
    pred_graph = model.predict(graph)
    df = pd.DataFrame()
    u, v = pred_graph.edges(etype=("team", "home", "team"))
    df["team"] = v
    df["opp"] = u
    df["pred"] = pred_graph.edata["pred"][("team", "home", "team")].detach()
    df = df.pivot_table(index="team", columns="opp", values="pred")
    if name_map is not None:
        df = df.rename(index=name_map, columns=name_map)
    if sort:
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index, axis=1)
    return df
