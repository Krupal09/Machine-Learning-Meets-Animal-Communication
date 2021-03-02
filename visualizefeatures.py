#!/net/projects/scratch/winter/valid_until_31_July_2021/hhameed/miniconda3/bin/python3

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("~/features2")

plt.scatter(df["bottleneck1"], df["bottleneck2"])
plt.savefig("ab.png")

plt.scatter(df["bottleneck1"], df["bottleneck3"])
plt.savefig("ac.png")

plt.scatter(df["bottleneck3"], df["bottleneck2"])
plt.savefig("cb.png")

