import pandas as pd


# csv Path:
df_path="//wsl.localhost/Ubuntu/home/gaiters/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/data/census.csv"
# load the dataset
df = pd.read_csv(df_path)

# Print the first 5 rows of the dataframe
df.head()