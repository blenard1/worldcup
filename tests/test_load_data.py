import pandas as pd
import pytest

from src.data.load_data import (
    CURRENT_PERFORMANCE_REQUIRED_COLUMNS,
    H2H_REQUIRED_COLUMNS,
    MATCHES_REQUIRED_COLUMNS,
    TACTICS_REQUIRED_COLUMNS,
    load_current_performance,
    load_h2h,
    load_matches,
    load_tactics,
)
from src.utils.config import SAMPLE_DATA_DIR


LOADER_CASES = [
    ("matches.csv", load_matches, MATCHES_REQUIRED_COLUMNS),
    ("team_tactics.csv", load_tactics, TACTICS_REQUIRED_COLUMNS),
    (
        "current_worldcup_performance.csv",
        load_current_performance,
        CURRENT_PERFORMANCE_REQUIRED_COLUMNS,
    ),
    ("h2h.csv", load_h2h, H2H_REQUIRED_COLUMNS),
]


@pytest.mark.parametrize("filename,loader,required_columns", LOADER_CASES)
def test_sample_files_exist_and_load(filename, loader, required_columns):
    assert (SAMPLE_DATA_DIR / filename).is_file()

    dataframe = loader()

    assert isinstance(dataframe, pd.DataFrame)
    assert not dataframe.empty
    assert list(required_columns) == [
        column for column in required_columns if column in dataframe.columns
    ]


@pytest.mark.parametrize("filename,loader,required_columns", LOADER_CASES)
def test_missing_required_column_raises_clear_error(
    tmp_path,
    filename,
    loader,
    required_columns,
):
    dataframe = pd.read_csv(SAMPLE_DATA_DIR / filename)
    missing_column = required_columns[0]
    broken_dataframe = dataframe.drop(columns=[missing_column])
    broken_path = tmp_path / filename
    broken_dataframe.to_csv(broken_path, index=False)

    with pytest.raises(ValueError, match=f"missing required columns: {missing_column}"):
        loader(broken_path)


def test_tactical_scores_are_in_expected_sample_range():
    tactics = load_tactics()
    score_columns = [
        column
        for column in TACTICS_REQUIRED_COLUMNS
        if column not in {"team", "formation", "defensive_block"}
    ]

    for column in score_columns:
        assert tactics[column].between(1, 10).all()
