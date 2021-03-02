
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('/net/projects/scratch/winter/valid_until_31_July_2021/hhameed/features1')

clustering = DBSCAN(eps=2, min_samples=50).fit(df)
clustering.labels_

pickle.dump(clustering, open('clusteringmodel1', 'wb'))
