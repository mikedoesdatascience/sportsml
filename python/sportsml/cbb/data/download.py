import urllib
from typing import List

import pandas as pd
import requests
import tqdm
from bs4 import BeautifulSoup

from .features import STATS_COLUMNS

columns_converters = {name: float for name in STATS_COLUMNS}


def download_years(years: List[int], path: str = None) -> pd.DataFrame:
    """
    Download game logs from `years` and save to disk.
    """
    if path is None:
        path = f"data/{min(years)}-{max(years)}_boxscores.json.gz"
    dfs = []
    for year in years:
        print(year)
        team_names = get_team_names(year)
        for team in tqdm.tqdm(team_names.values()):
            try:
                df = get_gamelogs(team, year)
            except ValueError:
                continue
            df["OPP_TEAM"] = df["OPP_TEAM"].map(team_names)
            if df is None:
                continue
            dfs.append(df)
    df = pd.concat(dfs)
    if path:
        df.reset_index(drop=True).to_json(
            path, orient="records", compression="gzip")
    return df


def get_team_names(year: int):
    """
    Download team names dictionary for year `year`
    """
    url = f"https://www.sports-reference.com/cbb/seasons/{year}-school-stats.html"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table")
    teams = {}
    for tr in table.findAll("tr"):
        tds = tr.findAll("td")
        for td in tds:
            try:
                team_id = td.find("a")["href"].split("/")[-2]
                team_name = td.find("a").text
                teams[team_name] = team_id
            except:
                pass
    return teams


def clean_basic_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning on a DataFrame of basic stats
    """
    game_data = df.droplevel(0, axis=1)[
        ["G", "Unnamed: 2_level_1", "Date", "Tm", "Opp"]
    ]
    game_data["OPP_TEAM"] = game_data["Opp"].iloc[:, 0]
    game_data = game_data.loc[:, ~game_data.columns.duplicated(keep="last")]
    game_data = game_data.rename(
        columns={"Tm": "PTS", "Opp": "OPP_PTS",
                 "Unnamed: 2_level_1": "HOME_AWAY"}
    )
    game_data["HOME_AWAY"] = game_data["HOME_AWAY"].fillna("H").replace({
        "@": "A"})
    return pd.concat(
        [game_data, df["School"], df["Opponent"].rename(
            columns=lambda x: f"OPP_{x}")],
        axis=1,
    )


def clean_advanced_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning on a DataFrame of advanced stats
    """
    df = df.droplevel(0, axis=1)
    df = df.rename(columns={"Unnamed: 2_level_1": "HOME_AWAY"})
    df["HOME_AWAY"] = df["HOME_AWAY"].fillna("H").replace({"@": "A"})
    return df


def get_basic_stats(team: str, year: int):
    """
    Download basic boxscore stats for team `team` in year `year`
    """
    url = f"https://www.sports-reference.com/cbb/schools/{team}/{year}-gamelogs.html"
    try:
        df = pd.read_html(url)[0]
    except urllib.error.HTTPError:
        return None
    df = clean_basic_df(df)
    mask = df["G"].astype(str).str.isnumeric() == True
    df = df[mask]
    df["TEAM"] = team
    return df.reset_index(drop=True)


def get_advanced_stats(team: str, year: int):
    """
    Download advanced boxscore stats for team `team` in year `year`
    """
    url = f"https://www.sports-reference.com/cbb/schools/{team}/{year}-gamelogs-advanced.html"
    try:
        df = pd.read_html(url)[0]
    except urllib.error.HTTPError:
        return None
    df = clean_advanced_df(df)
    mask = df["G"].astype(str).str.isnumeric() == True
    df = df[mask]
    df["TEAM"] = team
    return df.dropna(axis=1, how="all").reset_index(drop=True)


def get_gamelogs(team: str, year: int, date: str = None):
    """
    Download combined basic and advanced stats for team `team` from year `year`
    """
    basic = get_basic_stats(team, year)
    if basic is None:
        return None
    advanced = get_advanced_stats(team, year).drop(
        columns=["Opp", "Tm", "W/L"])
    df = pd.concat([basic, advanced], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.astype(columns_converters)
    df["DATE"] = df.pop("Date")
    df["SEASON"] = year
    if date is not None:
        df = df[df['DATE'] < date]
    return df


