from pathlib import Path

import polars as pl
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "penguins.csv"


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/download_data.py first")
def test_penguins_dataset_shape():
    df = pl.read_csv(DATA, null_values="NA")
    assert df.height == 344
    assert set(df["species"].unique()) == {"Adelie", "Chinstrap", "Gentoo"}
    assert df.drop_nulls().height == 333


def test_notebook_is_valid_marimo_app():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "notebooks" / "penguins.py"
    spec = importlib.util.spec_from_file_location("penguins_nb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "app")
