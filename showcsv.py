#展示csv文件的前几行

import pandas as pd

df = pd.read_csv('data/ancient/cn_wenyan.csv')

print(df.head())