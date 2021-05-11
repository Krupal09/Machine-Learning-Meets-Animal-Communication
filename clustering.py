from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



# extract features
df = pd.DataFrame( columns = ["filename"] + ["bottleneck" + str(_) for _ in range(nbottleneck)] )
for spectrogram in data:
    imagename = spectrogram["name"]
    spectrogram = spectrogram["spectrogram"]

    # if the spectrogram is not of width 194 units, don't run the iteration
    if spectrogram.shape[1] != 194:
        continue

    norm = np.linalg.norm(spectrogram)
    snippet = spectrogram / norm
    snippet = torch.reshape( snippet, (-1,) ).to(device)

    features = model.generatefeatures(snippet).detach().numpy()
    #features = [str(f.item()) for f in features]
    df = df.append( dict( zip( df.columns, [imagename] + list(features) ) ), ignore_index=True )

df.to_csv(folder + "/features")

# perform clustering
clustering = GaussianMixture(n_components=7, random_state=0).fit(df[[ "bottleneck" + str(_) for _ in range(nbottleneck) ]])

df["clusterlabels"] = clustering.means_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = [-1,1]
X, Y = np.meshgrid(r, r)
ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig(folder + '/gmmmixtures')

# write to file
df.to_csv(folder + "/features")

sys.stdout.close()
sys.stderr.close()
