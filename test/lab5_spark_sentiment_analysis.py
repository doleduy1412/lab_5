from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# 2. Load Data
data_path = "D:/Downloads/sentiments.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
initial_row_count = df.count()
df = df.dropna(subset=["sentiment"])
print(f"Loaded {initial_row_count} rows, after dropna: {df.count()} rows")

# 3. Build preprocessing pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(
    inputCol="filtered_words", outputCol="raw_features", numFeatures=10000
)
idf = IDF(inputCol="raw_features", outputCol="features")

# 4. Model
lr = LogisticRegression(
    maxIter=10, regParam=0.001, featuresCol="features", labelCol="label"
)

# Combine all stages
pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

# 5. Train / Test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_df)

# 6. Evaluate
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Spark Model Accuracy: {accuracy:.3f}")

spark.stop()
