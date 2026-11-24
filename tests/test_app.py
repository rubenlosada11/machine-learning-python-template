"""Unit tests for src/app.py. The real model is never downloaded: a fake classifier stands in for it."""
import pandas as pd
import pytest

import app


class FakeClassifier:
    """Mimics a transformers text-classification pipeline: list[str] -> list[{'label', 'score'}]."""

    def __init__(self, labels_by_text=None, default="5 stars", fail=False, drop_one=False):
        self.labels_by_text = labels_by_text or {}
        self.default = default
        self.fail = fail
        self.drop_one = drop_one
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if self.fail:
            raise RuntimeError("boom")
        out = [{"label": self.labels_by_text.get(t, self.default), "score": 0.87654} for t in texts]
        return out[:-1] if self.drop_one else out


def write_csv(path, rows, columns=("review_id", "rating", "review_text")):
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)
    return path


# --- label_to_stars / stars_to_sentiment ------------------------------------------------
@pytest.mark.parametrize("label, stars", [("1 star", 1), ("2 stars", 2), ("3 stars", 3),
                                          ("4 stars", 4), ("5 stars", 5), (" 5 Stars ", 5)])
def test_label_to_stars_parses_model_labels(label, stars):
    assert app.label_to_stars(label) == stars


@pytest.mark.parametrize("label", ["LABEL_4", "0 stars", "6 stars", "five stars", "", None, "4"])
def test_label_to_stars_rejects_unexpected_labels(label):
    with pytest.raises(ValueError):
        app.label_to_stars(label)


@pytest.mark.parametrize("stars, sentiment", [(1, "Negativo"), (2, "Negativo"), (3, "Neutral"),
                                              (4, "Positivo"), (5, "Positivo")])
def test_stars_to_sentiment_bands(stars, sentiment):
    assert app.stars_to_sentiment(stars) == sentiment


@pytest.mark.parametrize("stars", [0, 6, None, "4"])
def test_stars_to_sentiment_rejects_out_of_range(stars):
    with pytest.raises(ValueError):
        app.stars_to_sentiment(stars)


# --- input validation ---------------------------------------------------------------------
def test_load_reviews_missing_file(tmp_path):
    with pytest.raises(app.ReviewDataError, match="not found"):
        app.load_reviews(tmp_path / "nope.csv")


def test_load_reviews_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(app.ReviewDataError, match="empty"):
        app.load_reviews(path)


def test_load_reviews_header_only(tmp_path):
    path = write_csv(tmp_path / "header.csv", [])
    with pytest.raises(app.ReviewDataError, match="no reviews"):
        app.load_reviews(path)


def test_load_reviews_missing_columns(tmp_path):
    path = write_csv(tmp_path / "cols.csv", [(1, "Great")], columns=("review_id", "review_text"))
    with pytest.raises(app.ReviewDataError, match="rating"):
        app.load_reviews(path)


@pytest.mark.parametrize("bad_rating", [0, 6, "abc", None, 4.5])
def test_load_reviews_invalid_rating(tmp_path, bad_rating):
    path = write_csv(tmp_path / "rating.csv", [(1, 5, "Great"), (2, bad_rating, "Fine")])
    with pytest.raises(app.ReviewDataError, match="rating"):
        app.load_reviews(path)


def test_load_reviews_duplicated_ids(tmp_path):
    path = write_csv(tmp_path / "dup.csv", [(1, 5, "Great"), (1, 4, "Good")])
    with pytest.raises(app.ReviewDataError, match="Duplicated"):
        app.load_reviews(path)


# --- predict -------------------------------------------------------------------------------
def test_predict_keeps_rows_with_empty_text_without_classifying_them():
    df = pd.DataFrame({"review_id": [1, 2, 3, 4], "rating": [5, 1, 3, 4],
                       "review_text": ["Lovely", "   ", None, "Awful"]})
    fake = FakeClassifier({"Awful": "1 star"})
    out = app.predict(df, fake)

    assert out["review_id"].tolist() == [1, 2, 3, 4]
    assert fake.calls[0][0] == ["Lovely", "Awful"]            # blank/null texts never reach the model
    assert out["predicted_stars"].tolist()[0] == 5 and out["predicted_stars"].tolist()[3] == 1
    assert out.loc[[1, 2], "predicted_stars"].isna().all()
    assert out.loc[[1, 2], "sentiment"].isna().all()
    assert out["sentiment"].tolist()[3] == "Negativo"


def test_predict_uses_one_batched_call_and_preserves_input():
    df = pd.DataFrame({"review_id": [10, 20], "rating": [5, 3], "review_text": ["A", "B"]})
    original = df.copy()
    fake = FakeClassifier({"B": "3 stars"})
    out = app.predict(df, fake, batch_size=7)

    assert len(fake.calls) == 1
    assert fake.calls[0][1] == {"batch_size": 7, "truncation": True}
    pd.testing.assert_frame_equal(df, original)                 # input not mutated
    assert list(out.columns[: len(app.OUTPUT_COLUMNS)]) == app.OUTPUT_COLUMNS
    assert out["sentiment"].tolist() == ["Positivo", "Neutral"]
    assert out["model_score"].tolist() == [0.8765, 0.8765]


def test_predict_wraps_inference_errors():
    df = pd.DataFrame({"review_id": [1], "rating": [5], "review_text": ["A"]})
    with pytest.raises(app.ModelError, match="Inference failed"):
        app.predict(df, FakeClassifier(fail=True))


def test_predict_detects_missing_predictions():
    df = pd.DataFrame({"review_id": [1, 2], "rating": [5, 4], "review_text": ["A", "B"]})
    with pytest.raises(app.ModelError, match="1 predictions for 2"):
        app.predict(df, FakeClassifier(drop_one=True))


# --- end to end with the fake model --------------------------------------------------------
def test_run_writes_output_creating_missing_directory(tmp_path):
    src = write_csv(tmp_path / "reviews.csv", [(3, 5, "Great"), (1, 2, "Bad"), (2, 3, "Meh")])
    dest = tmp_path / "new" / "dir" / "out.csv"
    fake = FakeClassifier({"Bad": "2 stars", "Meh": "3 stars"})

    app.run(src, dest, classifier=fake)

    saved = pd.read_csv(dest)
    assert saved["review_id"].tolist() == [3, 1, 2]              # original order preserved
    assert list(saved.columns) == app.OUTPUT_COLUMNS
    assert saved["sentiment"].tolist() == ["Positivo", "Negativo", "Neutral"]
    assert saved["review_text"].tolist() == ["Great", "Bad", "Meh"]


def test_run_loads_model_once(tmp_path, monkeypatch):
    src = write_csv(tmp_path / "reviews.csv", [(i, 5, f"text {i}") for i in range(100)])
    loads = []
    monkeypatch.setattr(app, "load_classifier", lambda: loads.append(1) or FakeClassifier())

    app.run(src, tmp_path / "out.csv")

    assert loads == [1]


def test_main_returns_1_on_data_error(tmp_path, monkeypatch):
    monkeypatch.setattr(app.run, "__defaults__", (tmp_path / "missing.csv", tmp_path / "out.csv", None))
    assert app.main() == 1


def test_main_returns_2_when_model_cannot_load(tmp_path, monkeypatch):
    src = write_csv(tmp_path / "reviews.csv", [(1, 5, "Great")])
    monkeypatch.setattr(app.run, "__defaults__", (src, tmp_path / "out.csv", None))

    def broken_loader():
        raise app.ModelError("offline")

    monkeypatch.setattr(app, "load_classifier", broken_loader)
    assert app.main() == 2
    assert not (tmp_path / "out.csv").exists()
