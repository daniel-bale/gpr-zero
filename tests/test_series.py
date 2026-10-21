"""Category aggregation follows the supplied thesis artifact's dictionary set."""

import os
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gpr_zero.constants import STANDARD_CATEGORIES
from gpr_zero.gpr_index.series import category_medians


@pytest.mark.parametrize(
    "excluded",
    [
        "step2_gemini-2.5-pro_stable8cat_run8",
        "gemini-2.5-pro_stable8cat_run8",
    ],
)
def test_excluded_dictionary_does_not_shift_category_median(excluded):
    weekly = pd.DataFrame(
        {
            "week": pd.to_datetime(["2020-01-06"] * 4),
            "category": ["Military Escalation"] * 4,
            "dictionary": ["first", "second", "third", excluded],
            "gpr_index": [0.1, 0.3, 0.9, 0.95],
        }
    )
    original = weekly.copy(deep=True)
    assert weekly["gpr_index"].median() == pytest.approx(0.6)
    actual = category_medians(weekly)
    assert actual["gpr_index"].tolist() == [0.3]
    assert_frame_equal(weekly, original)


def test_category_aliases_share_one_normalized_median_per_week():
    weekly = pd.DataFrame(
        {
            "week": pd.to_datetime(["2020-01-06"] * 4 + ["2020-01-13"]),
            "category": [
                "Military Escalation",
                "Military Escalation (軍事升級)",
                "軍事升級",
                "Diplomatic Isolation",
                "軍事升級",
            ],
            "dictionary": ["first", "second", "third", "first", "first"],
            "gpr_index": [0.1, 0.3, 0.9, 0.4, 0.8],
        }
    )
    actual = category_medians(weekly)
    assert actual.groupby("category").size().to_dict() == {
        "Diplomatic Isolation": 1,
        "Military Escalation": 2,
    }
    assert actual.loc[
        actual["category"] == "Military Escalation", "gpr_index"
    ].tolist() == [0.3, 0.8]


def test_real_category_parquet_matches_every_supplied_profile_median():
    configured = os.environ.get("GPR_THESIS_PROFILE")
    if not configured:
        pytest.skip("Set GPR_THESIS_PROFILE to compare the external category artifacts")
    profile = Path(configured).expanduser()
    source = profile.parent / "indices" / "category_weekly.parquet"
    assert source.is_file(), f"Category weekly Parquet is unavailable: {source}"
    weekly = pd.read_parquet(source)
    selected = weekly.loc[
        (weekly["scope"] == "three_outlets")
        & (weekly["endpoint_policy"] == "legacy_end_midnight")
        & weekly["dictionary"].str.startswith("step2_")
    ].rename(columns={"category_share": "gpr_index"})
    actual = category_medians(selected).rename(columns={"gpr_index": "category_share"})
    actual = actual.sort_values(["week", "category"]).reset_index(drop=True)
    expected = (
        pd.read_csv(
            profile / "category_medians.csv",
            parse_dates=["week"],
            float_precision="round_trip",
        )
        .sort_values(["week", "category"])
        .reset_index(drop=True)
    )
    actual = actual[expected.columns]
    assert len(actual) == 3352
    assert set(actual["category"]) == set(STANDARD_CATEGORIES)
    assert actual.groupby("category").size().eq(419).all()
    assert_frame_equal(actual, expected, check_exact=True)
