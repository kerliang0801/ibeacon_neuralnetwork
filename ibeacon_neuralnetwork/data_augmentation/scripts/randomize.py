import pandas as pd


df = pd.read_csv('####.csv', header=None)
ds = df.sample(frac=1)
ds.to_csv('###.csv')