
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/sentimentdataset.csv"
VISUALS_DIR = "visuals"

sns.set_theme(style="whitegrid")


def load_and_clean() -> pd.DataFrame:
    """Loading the raw CSV and fixing problems found during exploration:
    whitespace in headers and string values, timestamp parsing, duplicates."""
    df = pd.read_csv(DATA_PATH)

    # Drop Kaggle's exported index columns ("Unnamed: 0" and friends)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Strip whitespace from column names AND string values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    #Parse timestamps into real datetimes (coerce -> bad rows become NaT)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    bad = df["Timestamp"].isna().sum()
    if bad:
        print(f"Dropping {bad} rows with unparseable timestamps")
        df = df.dropna(subset=["Timestamp"])

    #Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Rows: {before} -> {len(df)} after de-duplication")

    return df

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create the measures the raw data doesn't have. This is the
    'feature engineering' step: arithmetic and extraction, no ML."""
    # Engagement: one number summarizing interaction with a post
    df["engagement"] = df["Likes"].fillna(0) + df["Retweets"].fillna(0)

    # Time components pulled from the timestamp
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.day_name()
    df["month"] = df["Timestamp"].dt.to_period("M").astype(str)

    # Collapse the zoo of sentiment labels into three groups
    positive = {"Positive", "Joy", "Excitement", "Contentment", "Happy",
                "Happiness", "Love", "Gratitude", "Admiration", "Hopeful"}
    negative = {"Negative", "Anger", "Sadness", "Despair", "Fear",
                "Frustration", "Disgust", "Grief", "Bitterness", "Hate"}
    df["sentiment_group"] = df["Sentiment"].apply(
        lambda s: "Positive" if s in positive
        else "Negative" if s in negative
        else "Neutral"
    )
    print("Sentiment groups:\n", df["sentiment_group"].value_counts())
    return df

def analyze(df: pd.DataFrame) -> dict:
    
    results = {}

    # Q1: Which platform generates the most engagement per post?
    results["engagement_by_platform"] = (
        df.groupby("Platform")["engagement"].mean().sort_values(ascending=False)
    )

    # Q2: How does sentiment mix differ by platform?
    results["sentiment_mix"] = (
        pd.crosstab(df["Platform"], df["sentiment_group"], normalize="index")
        .round(3) * 100
    )

    # Q3: When are posts most engaging? (time-based analysis)
    results["engagement_by_hour"] = df.groupby("hour")["engagement"].mean()
    results["engagement_by_dow"] = (
        df.groupby("day_of_week")["engagement"].mean()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"])
    )
    results["monthly_sentiment"] = (
        df.groupby(["month", "sentiment_group"]).size().unstack(fill_value=0)
    )

    for name, table in results.items():
        print(f"\n===== {name} =====\n{table}")
    return results

if __name__ == "__main__":
    df = load_and_clean()
    df = add_derived_metrics(df)
    results = analyze(df)