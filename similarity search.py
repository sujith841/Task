from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder.master("local[*]").appName("Advanced Similarity Search").getOrCreate()

# Load the dataset (adjust the file path as necessary)
df = spark.read.csv("amazon.csv", header=True, inferSchema=True)

# Show the first few records to verify the dataset
df.show(5, truncate=False)

# Text Preprocessing: Tokenization and TF-IDF
tokenizer = Tokenizer(inputCol="about_product", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Build the pipeline
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])

# Fit and transform the data
model = pipeline.fit(df)
result = model.transform(df)

# Cache the result to speed up further operations
result.cache()

# Show the result with TF-IDF features
result.select("product_id", "about_product", "features").show(5, truncate=False)

# Define similarity metrics

# Cosine Similarity
def cosine_similarity(df):
    similarities = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                similarity = float(vec1.dot(vec2) / (Vectors.norm(vec1, 2) * Vectors.norm(vec2, 2)))
                similarities.append((i + 1, j + 1, similarity))
    
    return similarities

# Euclidean Distance
def euclidean_distance(df):
    distances = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                distance = float(np.linalg.norm(vec1.toArray() - vec2.toArray()))
                distances.append((i + 1, j + 1, distance))
    
    return distances

# Jaccard Similarity
def jaccard_similarity(df):
    similarities = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                intersection = float(np.sum(np.minimum(vec1.toArray(), vec2.toArray())))
                union = float(np.sum(np.maximum(vec1.toArray(), vec2.toArray())))
                similarity = intersection / union if union != 0 else 0
                similarities.append((i + 1, j + 1, similarity))
    
    return similarities

# Manhattan Distance
def manhattan_distance(df):
    distances = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                distance = float(np.sum(np.abs(vec1.toArray() - vec2.toArray())))
                distances.append((i + 1, j + 1, distance))
    
    return distances

# Hamming Distance (for binary data)
def hamming_distance(df):
    distances = []
    features = df.select("features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                distance = float(np.sum(np.abs(vec1.toArray() - vec2.toArray())) != 0)
                distances.append((i + 1, j + 1, distance))
    
    return distances


start_time = time.time()
cosine_sim = cosine_similarity(result)
cosine_time = time.time() - start_time
print("Cosine Similarity Time:", cosine_time)

start_time = time.time()
euclidean_dist = euclidean_distance(result)
euclidean_time = time.time() - start_time
print("Euclidean Distance Time:", euclidean_time)

start_time = time.time()
jaccard_sim = jaccard_similarity(result)
jaccard_time = time.time() - start_time
print("Jaccard Similarity Time:", jaccard_time)

start_time = time.time()
manhattan_dist = manhattan_distance(result)
manhattan_time = time.time() - start_time
print("Manhattan Distance Time:", manhattan_time)

start_time = time.time()
hamming_dist = hamming_distance(result)
hamming_time = time.time() - start_time
print("Hamming Distance Time:", hamming_time)

# Plot execution times
metrics = ['Cosine', 'Jaccard', 'Euclidean', 'Manhattan', 'Hamming']
times = [cosine_time, jaccard_time, euclidean_time, manhattan_time, hamming_time]

plt.bar(metrics, times)
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time of Similarity Metrics')
plt.show()

# Saving the results (similarities) to CSV files
cosine_sim_df = pd.DataFrame(cosine_sim, columns=["Product 1", "Product 2", "Cosine Similarity"])
cosine_sim_df.to_csv("cosine_similarity_results.csv", index=False)

euclidean_dist_df = pd.DataFrame(euclidean_dist, columns=["Product 1", "Product 2", "Euclidean Distance"])
euclidean_dist_df.to_csv("euclidean_distance_results.csv", index=False)

jaccard_sim_df = pd.DataFrame(jaccard_sim, columns=["Product 1", "Product 2", "Jaccard Similarity"])
jaccard_sim_df.to_csv("jaccard_similarity_results.csv", index=False)

manhattan_dist_df = pd.DataFrame(manhattan_dist, columns=["Product 1", "Product 2", "Manhattan Distance"])
manhattan_dist_df.to_csv("manhattan_distance_results.csv", index=False)

hamming_dist_df = pd.DataFrame(hamming_dist, columns=["Product 1", "Product 2", "Hamming Distance"])
hamming_dist_df.to_csv("hamming_distance_results.csv", index=False)

# Links to download the similarity results
cosine_sim_link = 'cosine_similarity_results.csv'
euclidean_dist_link = 'euclidean_distance_results.csv'
jaccard_sim_link = 'jaccard_similarity_results.csv'
manhattan_dist_link = 'manhattan_distance_results.csv'
hamming_dist_link = 'hamming_distance_results.csv'

print(f"Cosine Similarity results: {cosine_sim_link}")
print(f"Euclidean Distance results: {euclidean_dist_link}")
print(f"Jaccard Similarity results: {jaccard_sim_link}")
print(f"Manhattan Distance results: {manhattan_dist_link}")
print(f"Hamming Distance results: {hamming_dist_link}")