"""WeLoveReviews — sentiment classification of customer reviews (production script).

Reads data/raw/reviews.csv, classifies every review with the pre-trained Hugging Face
model nlptown/bert-base-multilingual-uncased-sentiment (1–5 stars), maps the stars to a
sentiment band and writes data/processed/reviews_with_sentiment.csv.

Run from the repository root:
    python src/app.py

Exit codes: 0 = success, 1 = input data error, 2 = model loading/inference error.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

# --- Configuration -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"  # pinned Hugging Face commit
BATCH_SIZE = 32
DEVICE = "auto"  # "auto" -> CUDA if available, otherwise CPU; or force "cpu" / "cuda"

REQUIRED_COLUMNS = ("review_id", "rating", "review_text")
OUTPUT_COLUMNS = [*REQUIRED_COLUMNS, "predicted_stars", "sentiment", "model_label", "model_score"]
VALID_RATINGS = {1, 2, 3, 4, 5}
SENTIMENT_BY_STARS = {1: "Negativo", 2: "Negativo", 3: "Neutral", 4: "Positivo", 5: "Positivo"}
LABEL_PATTERN = re.compile(r"^\s*([1-5])\s+stars?\s*$", re.IGNORECASE)

logger = logging.getLogger("welovereviews")


class ReviewDataError(ValueError):
    """The input CSV is missing, empty or does not have the expected content."""


class ModelError(RuntimeError):
    """The model could not be loaded or failed during inference."""


# --- Pure transformation logic -------------------------------------------------------
def label_to_stars(label: str) -> int:
    """Convert a model label such as '1 star' or '4 stars' into an integer 1–5."""
    match = LABEL_PATTERN.match(str(label))
    if not match:
        raise ValueError(f"Unrecognised model label: {label!r}")
    return int(match.group(1))


def stars_to_sentiment(stars: int) -> str:
    """1–2 -> Negativo, 3 -> Neutral, 4–5 -> Positivo."""
    try:
        return SENTIMENT_BY_STARS[stars]
    except KeyError:
        raise ValueError(f"Stars out of range 1–5: {stars!r}") from None


def validate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns, unique ids and valid ratings. Returns the frame unchanged."""
    if df.empty:
        raise ReviewDataError("The input CSV contains no reviews.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ReviewDataError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    if df["review_id"].isna().any():
        raise ReviewDataError("Some rows have no review_id.")
    duplicated = df.loc[df["review_id"].duplicated(), "review_id"].tolist()
    if duplicated:
        raise ReviewDataError(f"Duplicated review_id values: {duplicated[:10]}")

    ratings = pd.to_numeric(df["rating"], errors="coerce")
    invalid = ~ratings.isin(VALID_RATINGS)
    if invalid.any():
        bad = df.loc[invalid, ["review_id", "rating"]].head(10).to_dict("records")
        raise ReviewDataError(f"{int(invalid.sum())} rows have a rating outside 1–5: {bad}")
    return df


def load_reviews(path: Path) -> pd.DataFrame:
    """Read and validate the raw reviews CSV without modifying its content."""
    path = Path(path)
    if not path.is_file():
        raise ReviewDataError(f"Input file not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        raise ReviewDataError(f"Input file is empty: {path}") from None
    df = validate_reviews(df)
    df["rating"] = pd.to_numeric(df["rating"]).astype(int)
    return df


# --- Model ---------------------------------------------------------------------------
def resolve_device(device: str = DEVICE) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_classifier(model_name: str = MODEL_NAME, revision: str = MODEL_REVISION, device: str = DEVICE):
    """Load the Hugging Face text-classification pipeline once (weights go to the HF cache)."""
    try:
        from transformers import pipeline

        return pipeline("text-classification", model=model_name, revision=revision, device=resolve_device(device))
    except Exception as exc:  # network, missing revision, corrupt cache, incompatible backend...
        raise ModelError(f"Could not load model {model_name}@{revision}: {exc}") from exc


def predict(df: pd.DataFrame, classifier, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Return a copy of df with the prediction columns added.

    Rows with an empty/blank text are kept (traceability) with empty prediction fields.
    """
    result = df.copy()
    valid = result["review_text"].fillna("").astype(str).str.strip().ne("")
    if (~valid).any():
        logger.warning("%d reviews have no text and will not be classified: review_id %s",
                       int((~valid).sum()), result.loc[~valid, "review_id"].tolist()[:10])

    texts = result.loc[valid, "review_text"].astype(str).tolist()
    try:
        outputs = classifier(texts, batch_size=batch_size, truncation=True) if texts else []
    except Exception as exc:
        raise ModelError(f"Inference failed: {exc}") from exc
    if len(outputs) != len(texts):
        raise ModelError(f"The model returned {len(outputs)} predictions for {len(texts)} texts.")

    labels = [o["label"] for o in outputs]
    stars = [label_to_stars(label) for label in labels]
    result["model_label"] = pd.Series(pd.NA, index=result.index, dtype="object")
    result["model_score"] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    result["predicted_stars"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["sentiment"] = pd.Series(pd.NA, index=result.index, dtype="object")
    result.loc[valid, "model_label"] = labels
    result.loc[valid, "model_score"] = [round(float(o["score"]), 4) for o in outputs]
    result.loc[valid, "predicted_stars"] = stars
    result.loc[valid, "sentiment"] = [stars_to_sentiment(s) for s in stars]
    return result[OUTPUT_COLUMNS + [c for c in result.columns if c not in OUTPUT_COLUMNS]]


def save_predictions(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


# --- Orchestration ---------------------------------------------------------------------
def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH, classifier=None) -> pd.DataFrame:
    reviews = load_reviews(input_path)
    logger.info("Loaded %d reviews from %s", len(reviews), input_path)

    if classifier is None:
        logger.info("Loading model %s (revision %s)", MODEL_NAME, MODEL_REVISION[:10])
        classifier = load_classifier()

    t0 = time.perf_counter()
    enriched = predict(reviews, classifier)
    logger.info("Classified %d reviews in %.1f s", int(enriched["predicted_stars"].notna().sum()),
                time.perf_counter() - t0)

    if len(enriched) != len(reviews) or not enriched["review_id"].equals(reviews["review_id"]):
        raise ModelError("Output rows do not match input rows.")

    save_predictions(enriched, output_path)
    counts = enriched["sentiment"].value_counts()
    logger.info("Sentiment: %s", ", ".join(f"{k} {v} ({v / len(enriched):.1%})" for k, v in counts.items()))
    logger.info("Saved %s", output_path)
    return enriched


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide Hugging Face HTTP request logs
    try:
        run()
    except ReviewDataError as exc:
        logger.error("Input data error: %s", exc)
        return 1
    except ModelError as exc:
        logger.error("Model error: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
