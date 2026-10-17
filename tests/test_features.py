import pandas as pd
import pytest
from src.features import add_calendar_features, add_term, load_term_calendar


def test_summer_hour_uses_daylight_saving():
    # 2024-01-01 is a Monday; 00:00 UTC is 11:00 in Melbourne (UTC+11)
    out = add_calendar_features(pd.DataFrame({"hour_utc": pd.to_datetime(["2024-01-01 00:00"], utc=True)}))
    assert (out.loc[0, "hour"], out.loc[0, "dow"], out.loc[0, "hour_of_week"]) == (11, 0, 11)


def test_winter_hour_is_standard_time():
    # 2024-07-01 is a Monday; 00:00 UTC is 10:00 in Melbourne (UTC+10)
    out = add_calendar_features(pd.DataFrame({"hour_utc": pd.to_datetime(["2024-07-01 00:00"], utc=True)}))
    assert (out.loc[0, "hour"], out.loc[0, "hour_of_week"]) == (10, 10)


def _calendar(tmp_path, text):
    """Write a small calendar file and load it for 1 to 10 January 2024."""
    f = tmp_path / "cal.csv"
    f.write_text(text)
    return load_term_calendar(f, first="2024-01-01", last="2024-01-10")


def test_later_rows_override_earlier_rows(tmp_path):
    cal = _calendar(tmp_path, "start,end,period\n2024-01-01,2024-01-10,teaching\n"
                              "2024-01-05,2024-01-05,closure\n")
    assert [cal["2024-01-04"], cal["2024-01-05"], cal["2024-01-06"]] == ["teaching", "closure", "teaching"]


def test_unknown_period_label_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _calendar(tmp_path, "start,end,period\n2024-01-01,2024-01-10,holidays\n")


def test_uncovered_day_is_rejected(tmp_path):
    cal = _calendar(tmp_path, "start,end,period\n2024-01-01,2024-01-05,teaching\n")
    df = add_calendar_features(pd.DataFrame({"hour_utc": pd.to_datetime(["2024-01-08 00:00"], utc=True)}))
    with pytest.raises(ValueError):
        add_term(df, cal)


def test_local_date_decides_the_period(tmp_path):
    # 4 Jan 20:00 UTC is 07:00 on 5 Jan in Melbourne (UTC+11), which is the closure day
    cal = _calendar(tmp_path, "start,end,period\n2024-01-01,2024-01-10,teaching\n"
                              "2024-01-05,2024-01-05,closure\n")
    df = add_calendar_features(pd.DataFrame({"hour_utc": pd.to_datetime(["2024-01-04 20:00"], utc=True)}))
    assert add_term(df, cal).loc[0, "period"] == "closure"