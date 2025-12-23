import pandas as pd
import matplotlib.pyplot as plt
import os


BASE_PATH ="C:\movie_project" 

DATA_PATH = os.path.join(BASE_PATH, "data", "movies_cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs", "graphs")


os.makedirs(OUTPUT_PATH, exist_ok=True)


df = pd.read_csv(DATA_PATH)

print("EDA started on cleaned dataset")
print("Dataset shape:", df.shape)


plt.figure(figsize=(8,5))
df["Genre"].value_counts().plot(kind="bar")
plt.title("Movies Count by Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "genre_count.png"))
plt.close()


plt.figure(figsize=(8,5))
df["Profitability"].hist(bins=20)
plt.title("Profitability Distribution")
plt.xlabel("Profitability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "profitability_distribution.png"))
plt.close()


plt.figure(figsize=(8,5))
plt.scatter(df["Rotten Tomatoes %"], df["Profitability"])
plt.title("Rotten Tomatoes Score vs Profitability")
plt.xlabel("Rotten Tomatoes %")
plt.ylabel("Profitability")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "rt_vs_profitability.png"))
plt.close()


plt.figure(figsize=(8,5))
plt.scatter(df["Audience score %"], df["Profitability"])
plt.title("Audience Score vs Profitability")
plt.xlabel("Audience Score %")
plt.ylabel("Profitability")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "audience_vs_profitability.png"))
plt.close()

print("✅ EDA completed successfully!")
print("📁 Graphs saved in:", OUTPUT_PATH)
