import pandas as pd


df = pd.read_csv("C:\movie_project\data\movies.csv")


print("Original Dataset Shape:", df.shape)


df["Worldwide Gross"] = df["Worldwide Gross"].astype(str)
df["Worldwide Gross"] = df["Worldwide Gross"].str.replace("$", "")
df["Worldwide Gross"] = df["Worldwide Gross"].str.replace(",", "")
df["Worldwide Gross"] = pd.to_numeric(df["Worldwide Gross"], errors="coerce")

df["Audience score %"] = pd.to_numeric(df["Audience score %"], errors="coerce")
df["Rotten Tomatoes %"] = pd.to_numeric(df["Rotten Tomatoes %"], errors="coerce")

df_cleaned = df.dropna(subset=[
    "Audience score %",
    "Rotten Tomatoes %",
    "Worldwide Gross",
    "Profitability"
])

print("Cleaned Dataset Shape:", df_cleaned.shape)


df_cleaned.to_csv("movies_cleaned.csv", index=False)

print("✅ Data cleaning completed successfully!")
print("📁 Cleaned file saved as: movies_cleaned.csv")
