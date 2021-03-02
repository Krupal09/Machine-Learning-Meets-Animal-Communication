#!/net/projects/scratch/winter/valid_until_31_July_2021/hhameed/miniconda3/bin/python3

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('/net/projects/scratch/winter/valid_until_31_July_2021/hhameed/lucidmonkeys/features2')

clustering = KMeans(n_clusters=3, random_state=0).fit(df[["bottleneck1", "bottleneck2", "bottleneck3"]])

df["label"] = clustering.labels_

df.to_csv("labeledfeatures")
#pickle.dump(clustering, open('/net/projects/scratch/winter/valid_until_31_July_2021/hhameed/lucidmonkeys/clusteringmodel1', 'wb'))
