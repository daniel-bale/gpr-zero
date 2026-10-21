"""Weekly spike events and the thesis workflows that consume them.

Detection groups adjacent *rows*, even when weeks are missing. Stored start/end
dates identify week starts; article sampling includes the final week's six
remaining days. Peak validation and summary overlap deliberately remain distinct
policies because the thesis uses both.
"""

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import pandas as pd

Event = Mapping[str, Any]


def _week_window(event: Event) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = event["start_date"] if "start_date" in event else event["start"]
    end = event["end_date"] if "end_date" in event else event["end"]
    return pd.Timestamp(start), pd.Timestamp(end)


def article_window(event: Event) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive article dates for an event, including its entire final week."""
    start, end = _week_window(event)
    return start, end + timedelta(days=6)


def events_overlap(left: Event, right: Event) -> bool:
    """Compare inclusive stored week starts, as in newspaper spike uniqueness."""
    left_start, left_end = _week_window(left)
    right_start, right_end = _week_window(right)
    return not (left_end < right_start or left_start > right_end)


def _detect_groups(frame: pd.DataFrame, value_column: str):
    # Pandas' sample standard deviation (ddof=1), strict threshold and the
    # original is_spike mutation are all observable thesis behavior.
    values = frame[value_column]
    threshold = values.mean() + values.std()
    frame["is_spike"] = values > threshold
    groups = []
    rows = []
    for _, row in frame.iterrows():
        if row["is_spike"]:
            rows.append(row)
        elif rows:
            groups.append(pd.DataFrame(rows))
            rows = []
    if rows:
        groups.append(pd.DataFrame(rows))
    return groups, threshold


def detect_weekly_events(
    frame: pd.DataFrame, value_column: str
) -> tuple[list[dict], float]:
    """Detect unified events from a ``date`` column, retaining the CSV schema."""
    groups, threshold = _detect_groups(frame, value_column)
    events = []
    for group in groups:
        # idxmax selects the first tied peak, not the last one.
        peak = group.loc[group[value_column].idxmax()]
        events.append(
            {
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "peak_date": peak["date"],
                "peak_value": peak[value_column],
                "duration_weeks": len(group),
                "mean_spike_value": group[value_column].mean(),
            }
        )
    return events, threshold


def detect_newspaper_events(frame: pd.DataFrame) -> list[dict]:
    """Detect newspaper events from indexed ``median_scaled`` observations.

    This adapter retains row-order endpoints and the historical ``weeks`` key.
    Unified exports instead use date extrema and ``duration_weeks``.
    """
    groups, _ = _detect_groups(frame, "median_scaled")
    events = []
    for group in groups:
        peak_position = group["median_scaled"].argmax()
        peak_date = group.index[peak_position]
        events.append(
            {
                "start_date": group.index[0],
                "end_date": group.index[-1],
                "peak_value": group.iloc[peak_position]["median_scaled"],
                "peak_date": peak_date,
                "weeks": len(group),
            }
        )
    return events


def compare_peak_events(new_events, old_frame, tolerance_days=7) -> list[dict]:
    """Greedily validate peaks against the first unused thesis event in range."""
    old_events = [
        {
            "start_date": pd.to_datetime(row["start_date"]),
            "end_date": pd.to_datetime(row["end_date"]),
            "peak_date": pd.to_datetime(row["peak_date"]),
            "peak_value": row.get("peak_gpr_share"),
        }
        for _, row in old_frame.iterrows()
    ]
    results = []
    matched_old = set()
    for new in new_events:
        for index, old in enumerate(old_events):
            if index in matched_old:
                continue
            distance = abs((new["peak_date"] - old["peak_date"]).days)
            if distance <= tolerance_days:
                matched_old.add(index)
                results.append(
                    {
                        "status": "MATCHED",
                        "new_peak_date": new["peak_date"],
                        "old_peak_date": old["peak_date"],
                        "peak_date_diff_days": distance,
                        "new_peak_value": new["peak_value"],
                        "old_peak_value": old["peak_value"],
                        "new_duration": new["duration_weeks"],
                        "old_duration": (old["end_date"] - old["start_date"]).days // 7
                        + 1,
                    }
                )
                break
        else:
            results.append(
                {
                    "status": "NEW_ONLY",
                    "new_peak_date": new["peak_date"],
                    "new_peak_value": new["peak_value"],
                    "new_duration": new["duration_weeks"],
                }
            )
    for index, old in enumerate(old_events):
        if index not in matched_old:
            results.append(
                {
                    "status": "OLD_ONLY",
                    "old_peak_date": old["peak_date"],
                    "old_peak_value": old["peak_value"],
                    "old_duration": (old["end_date"] - old["start_date"]).days // 7 + 1,
                }
            )
    return results


def _summary_dates(event: Event) -> pd.DatetimeIndex:
    start, end = _week_window(event)
    # Summary CSVs use the peak to identify a single-week record. Keep this
    # exception local; neither article sampling nor newspaper comparison does.
    if start == end:
        start = end = pd.Timestamp(event["peak_date"])
    return pd.date_range(start, end, freq="D")


def select_overlapping_events(
    all_events: pd.DataFrame, selected_events: pd.DataFrame
) -> pd.DataFrame:
    """Select summary rows using the thesis single-week peak/interval policy."""
    selected = {
        date for _, row in selected_events.iterrows() for date in _summary_dates(row)
    }
    keep = [
        any(date in selected for date in _summary_dates(row))
        for _, row in all_events.iterrows()
    ]
    return all_events.loc[keep].copy()


def unique_events_by_newspaper(spike_results: Mapping[str, list]) -> dict:
    """Return legacy newspaper records with no overlap in any other newspaper."""
    events = {
        newspaper: [
            {
                "start": _week_window(event)[0],
                "end": _week_window(event)[1],
                "peak_value": event["peak_value"],
            }
            for event in spikes
        ]
        for newspaper, spikes in spike_results.items()
    }
    return {
        newspaper: [
            event
            for event in spikes
            if not any(
                events_overlap(event, other)
                for other_newspaper, others in events.items()
                if other_newspaper != newspaper
                for other in others
            )
        ]
        for newspaper, spikes in events.items()
    }
