import datetime
import os
import pytz

import httpx


class OddsAPIClient(object):
    def __init__(self, api_key: str = None, version: int = 4):
        self.host = f"https://api.the-odds-api.com/v{version}"
        if api_key is None:
            api_key = os.getenv("ODDS_API_KEY")
        self.api_key = api_key

    def get(self, path: str, params: dict = {}):
        params.update({"api_key": self.api_key})
        resp = httpx.get(f"{self.host}/{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def sports(self):
        return self.get("sports")

    def odds(
        self,
        sport: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso",
        convert_timezone: str = "US/Eastern",
    ):
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        odds = self.get(f"sports/{sport}/odds", params)
        for odd in odds:
            dt = datetime.datetime.fromisoformat(odd["commence_time"])
            odd["commence_time"] = dt.astimezone(
                pytz.timezone(convert_timezone)
            ).isoformat()
        return odds
