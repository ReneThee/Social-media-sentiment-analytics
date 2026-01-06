import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("~/Documents/Sentimental_analysis/sentimentdataset.csv")

# Check data
print(df.head())
print(df.info())
print(df.describe(include="all"))

# Cleaning the data
df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

text_cols = ["Sentiment", "Platform", "Country", "User"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df["Retweets"] = df["Retweets"].fillna(0)
df["Likes"] = df["Likes"].fillna(0)

df["Total_Engagement"] = df["Likes"] + df["Retweets"]

# Create Hour from Timestamp (needed later)
df["Hour"] = df["Timestamp"].dt.hour

# Univariate Analysis
plt.figure()
sns.countplot(data=df, x="Platform")
plt.title("Posts by Platform")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.histplot(df["Total_Engagement"], bins=30, kde=True)
plt.title("Distribution of Total Engagement")
plt.tight_layout()
plt.show()

# Bivariate Analysis
plt.figure()
sns.boxplot(data=df, x="Platform", y="Total_Engagement")
plt.title("Engagement by Platform")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Time-based analysis
hourly_posts = df.groupby("Hour").size()

plt.figure()
hourly_posts.plot(kind="line")
plt.title("Posting Activity by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.show()

sentiment_time = df.groupby(["Hour", "Sentiment"]).size().unstack(fill_value=0)
sentiment_time.plot(kind="line", figsize=(10, 5))
plt.title("Sentiment Trend Over the Day")
plt.xlabel("Hour")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.show()

# Country analysis
country_engagement = (
    df.groupby("Country")["Total_Engagement"]
    .mean()
    .sort_values(ascending=False)
)

print("Average engagement by country:")
print(country_engagement)

# Hashtag analysis
hashtags = df["Hashtags"].astype(str).str.split().explode()
top_hashtags = hashtags.value_counts().head(10)

print("Top 10 hashtags:")
print(top_hashtags)
