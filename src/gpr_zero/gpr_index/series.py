"""Aggregate weekly dictionary observations into the selected thesis series."""

import pandas as pd

from gpr_zero.constants import CATEGORY_MAPPING

# The supplied thesis profile's category_medians.csv selects 39 dictionaries.
# Step2 batch filenames omit the prefix used by the all-dictionaries inventory.
_EXCLUDED_CATEGORY_DICTIONARIES = frozenset(
    {
        "step2_gemini-2.5-pro_stable8cat_run8",
        "gemini-2.5-pro_stable8cat_run8",
    }
)


def category_medians(weekly: pd.DataFrame) -> pd.DataFrame:
    """Return category/week/gpr_index medians for the thesis category selection.

    Input rows contain week, category, dictionary and gpr_index, with one value
    per dictionary/category/week. Article cohort and timestamp filtering belong
    to the producer. Keep the profile's run-8 exclusion here so calculation and
    verification use the same selection, regardless of dictionary filename form.
    Category aliases follow the category workflow's existing normalization.
    """
    selected = weekly.loc[
        ~weekly["dictionary"].isin(_EXCLUDED_CATEGORY_DICTIONARIES),
        ["week", "category", "gpr_index"],
    ].copy()
    selected["category"] = selected["category"].replace(CATEGORY_MAPPING)
    return selected.groupby(["category", "week"])["gpr_index"].median().reset_index()
