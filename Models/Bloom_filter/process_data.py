import pandas as pd
import numpy as np
import random

df = pd.read_csv('crime.csv')
size = 3000
grid = np.zeros([size, size])

df = df[df['Lat']!=-1.0]
df = df[['Lat', 'Long', 'Location']].dropna()
lats = df['Lat'].values
longs = df['Long'].values
locations = df['Location'].values.tolist()

print(df.shape)