import pandas as pd
import os
import numpy as np
from visualize import plotResnet

df = pd.read_csv('results/SRResNet_baseline/hisResnetData.csv', index_col=0)
print(df)
df.replace(0, np.nan, inplace=True)
print(df)
df.to_csv('results/SRResNet_baseline/hisResnetData.csv')
# plotResnet('results/SRResNet_baseline/hisResnetData.csv', 'figure/')
