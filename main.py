import pandas as pd

# Lecture du fichier
df = pd.read_parquet('data/bronze_taxi.parquet')

# Vérification
print(df.head())
print(df.info())