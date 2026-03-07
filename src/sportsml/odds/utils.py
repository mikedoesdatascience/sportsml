import datetime

import pandas as pd


def odds_to_dataframe(odds):
    df = pd.DataFrame(odds).explode("bookmakers").reset_index(drop=True)
    bookmakers = (
        pd.DataFrame(df["bookmakers"].tolist())
        .rename(columns={"key": "bookmaker"})
        .drop(columns=["title", "last_update"])
    )
    df = pd.concat([df.drop(columns=["bookmakers"]), bookmakers], axis=1)

    df = df.explode("markets").reset_index(drop=True)
    markets = pd.DataFrame(df["markets"].tolist()).rename(columns={"key": "market"})
    df = pd.concat([df.drop(columns=["markets"]), markets], axis=1)

    df = pd.concat([df.drop(columns=["outcomes"]), df.apply(format_row, axis=1)], axis=1)

    df["market"] = df["market"].replace(
        {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
    )

    return df


def consensus_odds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(["commence_time", "home_team", "away_team", "market"])[
        [
            'away_moneyline',
            'away_spread_odds',
            'home_moneyline',
            'home_spread_odds',
            'over_odds',
            'spread_line',
            'total_line',
            'under_odds'
        ]
    ].mean().reset_index().assign(last_update=df['last_update'].max())

    dfs = [
        df[df['market'] == market].dropna(how='all', axis=1).drop(columns=['market', 'last_update']).set_index(['home_team', 'away_team'])
        for market in ['moneyline', 'spread', 'total']
    ]

    df = pd.concat(dfs, axis=1).reset_index().assign(last_update=df['last_update'].max())

    df = df.loc[:, ~df.columns.duplicated()]

    df = df[(pd.to_datetime(df['commence_time']) - pd.to_datetime(df['last_update'])).dt.days < 7]

    return df


def format_row(row):
    if row["market"] == "h2h":
        return pd.Series(format_h2h(row))
    elif row["market"] == "spreads":
        return pd.Series(format_spreads(row))
    elif row["market"] == "totals":
        return pd.Series(format_totals(row))
    else:
        raise ValueError(f"found unexpected market: {row['market']}")


def format_totals(row):
    res = {}
    for val in row["outcomes"]:
        res[f"{val['name'].lower()}_odds"] = val["price"]
        res["total_line"] = val["point"]
    return res


def format_spreads(row):
    res = {}
    for outcome in row["outcomes"]:
        if outcome["name"] == row["home_team"]:
            res["home_spread_odds"] = outcome["price"]
            res["spread_line"] = outcome["point"]
        if outcome["name"] == row["away_team"]:
            res["away_spread_odds"] = outcome["price"]
    return res


def format_h2h(row):
    res = {}
    for outcome in row["outcomes"]:
        if outcome["name"] == row["home_team"]:
            res["home_moneyline"] = outcome["price"]
        if outcome["name"] == row["away_team"]:
            res["away_moneyline"] = outcome["price"]
    return res
