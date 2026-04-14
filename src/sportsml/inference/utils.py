import pandas as pd


def long_to_wide(
    preds: pd.DataFrame,
    prob: bool = False,
    team_name_map: dict[int, str] = None,
    sorted: bool = True,
):
    preds = preds.pivot_table(
        values="prob" if prob else "preds", index="team", columns="opp"
    )
    if team_name_map:
        preds = preds.rename(index=team_name_map, columns=team_name_map)
    if sorted:
        preds = preds.loc[
            preds.mean(axis=0).sort_values().index,
            preds.mean(axis=0).sort_values().index,
        ]
    return preds