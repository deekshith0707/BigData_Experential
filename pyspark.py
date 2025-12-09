# =====================================================================
# 🔥 BIG DATA ANALYTICS PROJECT — TWITTER SENTIMENT ANALYSIS (COLAB)
# =====================================================================

# ---------------------------
# 1. INSTALL DEPENDENCIES
# ---------------------------


import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import re

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import udf

# =====================================================================
# 2. CREATE SAMPLE DATASET (OR REPLACE WITH YOUR OWN CSV)
# =====================================================================

data = [
    [1, "alice", "2025-01-05 10:00:20", "I love spark and big data! #spark #bigdata", 40],
    [2, "bob", "2025-01-05 10:01:05", "Spark is too complex sometimes. #confused", 15],
    [3, "carol", "2025-01-05 10:03:10", "Data science is amazing! #datascience", 60],
    [4, "dave", "2025-01-05 10:05:45", "I hate slow processing systems. #slow", 2],
    [5, "ellen", "2025-01-05 10:06:30", "Big data is overwhelming today. #bigdata", 8],
    [6, "john", "2025-01-05 10:07:15", "Great performance on our new spark cluster! #performance", 100],
    [7, "bob", "2025-01-05 10:09:00", "I don't like the new UI. #badux", 5],
    [8, "carol", "2025-01-05 10:11:25", "Cloud scaling works perfectly! #cloud #scale", 75],
    [9, "dave", "2025-01-05 10:12:40", "This update ruined everything. #fail", 3],
    [10, "ellen", "2025-01-05 10:14:00", "Love the improvements to data pipelines! #pipelines", 55],
]

df_pd = pd.DataFrame(data, columns=["tweet_id", "user", "created_at", "tweet", "likes"])
df_pd.to_csv("tweets.csv", index=False)

print("Dataset saved as tweets.csv")
df_pd.head()

# =====================================================================
# 3. START SPARK SESSION
# =====================================================================

spark = (
    SparkSession.builder
    .appName("ColabTwitterSentiment")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print("\n🔥 Spark session started!")

# =====================================================================
# 4. LOAD DATASET USING PYSPARK
# =====================================================================

df = spark.read.csv("tweets.csv", header=True, inferSchema=True)

print("\n=== RAW DATA LOADED ===")
df.show(5, truncate=False)
print("Total rows:", df.count())
df.printSchema()

# =====================================================================
# 5. TEXT CLEANING
# =====================================================================

def clean(text):
    if text is None:
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)       # remove URLs
    text = re.sub(r"[^A-Za-z0-9#@ ]+", " ", text)      # remove special chars
    text = re.sub(r"\s+", " ", text)                   # normalize spaces
    return text.lower().strip()

clean_udf = udf(clean, StringType())
df = df.withColumn("clean_tweet", clean_udf("tweet"))

print("\n=== CLEANED TWEETS ===")
df.select("tweet", "clean_tweet").show(5, truncate=False)

# =====================================================================
# 6. SENTIMENT ANALYSIS
# =====================================================================

def get_polarity(text):
    if not text:
        return 0.0
    return float(TextBlob(text).sentiment.polarity)

def label_sentiment(p):
    if p > 0.1:
        return "positive"
    elif p < -0.1:
        return "negative"
    else:
        return "neutral"

polarity_udf = udf(get_polarity, DoubleType())
sentiment_udf = udf(label_sentiment, StringType())

df = df.withColumn("polarity", polarity_udf("clean_tweet"))
df = df.withColumn("sentiment", sentiment_udf("polarity"))

print("\n=== SENTIMENT RESULTS ===")
df.select("clean_tweet", "polarity", "sentiment").show(10, truncate=False)

df.cache()

# =====================================================================
# 7. SENTIMENT DISTRIBUTION
# =====================================================================

sentiment_dist = (
    df.groupBy("sentiment")
      .agg(F.count("*").alias("count"))
      .orderBy(F.desc("count"))
)

print("\n=== SENTIMENT DISTRIBUTION ===")
sentiment_dist.show()

# =====================================================================
# 8. HASHTAG EXTRACTION
# =====================================================================

df = df.withColumn("words", F.split(F.col("clean_tweet"), " "))

df_hashtags = df.withColumn(
    "hashtag",
    F.explode(
        F.expr("filter(words, x -> x like '#%')")
    )
)

print("\n=== HASHTAGS EXTRACTED ===")
df_hashtags.select("clean_tweet", "hashtag", "sentiment").show(10, truncate=False)

hashtag_stats = (
    df_hashtags.groupBy("hashtag", "sentiment")
               .agg(F.count("*").alias("count"))
               .orderBy(F.desc("count"))
)

print("\n=== HASHTAG STATS ===")
hashtag_stats.show(20, truncate=False)

# =====================================================================
# 9. TIME-BASED SENTIMENT TREND
# =====================================================================

df = df.withColumn("timestamp", F.to_timestamp("created_at"))

df_time = (
    df.withColumn("minute", F.date_format("timestamp", "HH:mm"))
      .groupBy("minute", "sentiment")
      .agg(F.count("*").alias("count"))
      .orderBy("minute")
)

print("\n=== SENTIMENT TREND BY TIME ===")
df_time.show(20, truncate=False)

# =====================================================================
# 10. SAVE RESULTS TO PARQUET (BIG DATA FORMAT)
# =====================================================================

sentiment_dist.write.mode("overwrite").parquet("/content/sentiment_dist")
hashtag_stats.write.mode("overwrite").parquet("/content/hashtag_stats")
df_time.write.mode("overwrite").parquet("/content/time_trend")

print("\n📁 Saved output in Parquet format inside /content folder.")

# =====================================================================
# 11. VISUALIZATION — SENTIMENT BAR CHART
# =====================================================================

sent_pd = sentiment_dist.toPandas()

plt.figure()
plt.bar(sent_pd["sentiment"], sent_pd["count"], color=["green", "gray", "red"])
plt.title("Sentiment Distribution of Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =====================================================================
# 12. STOP SPARK SESSION
# =====================================================================
spark.stop()
print("\n✔️ Spark session stopped.")
