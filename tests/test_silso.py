import pandas as pd
from src.data.api.silso.silso import get_universal_silso_data


def test_get_universal_silso_data_parses_expected_columns(monkeypatch):
    sample_html = """
    <html>
    <body>
    Line format [character position]:
    [1-4] Year
    [6-7] Month
    [9-12] Date_Count
    ------------------
    <a href="DATA/SN_m_tot_V4.0.txt">data</a>
    </body>
    </html>
    """

    class DummyResponse:
        status_code = 200
        text = sample_html

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "src.data.api.silso.silso.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )

    captured = {}

    def fake_read_fwf(url, colspecs, header, names, na_values):
        captured["url"] = url
        captured["colspecs"] = colspecs
        captured["names"] = names
        return pd.DataFrame()

    monkeypatch.setattr("src.data.api.silso.silso.pd.read_fwf", fake_read_fwf)

    df = get_universal_silso_data()

    assert df is not None
    assert captured["url"].endswith("DATA/SN_m_tot_V4.0.txt")
    assert captured["names"] == ["year", "month", "date_count"]
    assert captured["colspecs"] == [(0, 4), (5, 7), (8, 12)]
