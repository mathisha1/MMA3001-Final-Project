import pandas as pd
from src.features import add_calendar_features


def test_summer_hour_uses_daylight_saving():
    # 2024-01-01 is a Monday; 00:00 UTC is 11:00 in Melbourne (UTC+11)
    out = add_calendar_features(pd.DataFrame({"hour_utc": pd.to_datetime(["2024-01-01 00:00"], utc=True)}))
    assert (out.loc[0, "hour"], out.loc[0, "dow"], out.loc[0, "hour_of_week"]) == (11, 0, 11)


def test_winter_hour_is_standard_time():
    # 2024-07-01 is a Monday; 00:00 UTC is 10:00 in Melbourne (UTC+10)
    out = add_calendar_features(pd.DataFrame({"hour_utc": pd.to_datetime(["2024-07-01 00:00"], utc=True)}))
    assert (out.loc[0, "hour"], out.loc[0, "hour_of_week"]) == (10, 10)