#!/usr/bin/env python3

"""
Module: clustering.py

Clustering for convolutional autoencoder :
* Iterate through the whole dataset(both training and validation) passing them
  through only encoder to get bottleneck vectors for each data point.
* Cluster these bottleneck vectors using unsupervised clustering methods like
  Kmeans or GaussianMixture modeling.

Authors: Krupal, Rachael

"""

import os
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#local system hangs while processing GMM so commented out for now
from sklearn.mixture import GaussianMixture
import matplotlib as plt

parser = argparse.ArgumentParser()

""" Directory parameters """

parser.add_argument(
    "--clustering_dir",
    type=str,
    help="The directory where the clustering outputs will be stored.",
)

""" Clustering parameters """

#parser.add_argument(
#    "--clustering",
#    action="store_true",
#    help="Performing clustering right after ae training",
#)

parser.add_argument(
    "--calc_optimal_num_clusters", type=str, default='gap', help="use elbow or gap statistics to find the optimal number of clusters",
)

parser.add_argument(
    "--num_clusters", type=int, default=4, help="The number of clusters to cluster all the data points."
)

ARGS = parser.parse_args()

class Elbow:
    """ Finding optimal number of clusters, by plotting elbow curve on KMeans scores (negative of the K-means objective)"""

    def __init__(self, num_clusters=range(1,30), init='k-means++', random_state=0, max_iter=300):
        self._num_clusters = num_clusters
        self._init = init
        self._random_state = random_state
        self._max_iter = max_iter

    def calc(self, inputs):
        kmeans = [KMeans(
            n_clusters=i,
            init = self._init,
            random_state = self._random_state,
            max_iter = self._max_iter,
        ) for i in self._num_clusters]
        scores = [kmeans[i].fit(inputs).score(inputs) for i in range(len(kmeans))]
        return scores

    def elbow_plot(self, scores, path):
        plt.plot(self._num_clusters, scores)
        plt.xlabel('Number of clusters')
        plt.ylabel('Score (negative of the K-means objective)')
        plt.title('Elbow curve')
        plt.savefig(os.path.join(path, 'ElbowCurve.png'))


class GapStatistics:
    def __init__(self):

if __name__ == '__main__':

    # load the trained model

    # collect embeddings for all data points
    file_names = []
    bottleneck_outputs = []

    with torch.no_grad():
        for i, (input_specs, label) in enumerate(dataloader):
            # remove file path to have only file name, ex : ['path/to/directory/file_1.wav']
            file_name = str(label['file_name'])[::-1]  # reverse string
            file_name = file_name.split("/")[0]
            file_name = file_name[::-1].split("'")[0]  # align it back to the right order
            file_names.append(file_name)
            # log.info("File-name : {}".format(file_name))

            input_specs = input_specs.to(device=ARGS.device)
            bottleneck_output = model.encoder(input_specs)
            bottleneck_output = np.reshape(bottleneck_output.detach().numpy(), newshape=(-1))
            bottleneck_outputs.append(bottleneck_output)

    if ARGS.calc_optimal_num_clusters == 'elbow':
        """ Finding optimal number of clusters, by using elbow curve (below are default values)"""
        elbow = Elbow(
            num_clusters=range(1, 30),
            init='k-means++',
            random_state=0,
            max_iter=300
        )
        scores = elbow.calc(bottleneck_output)
        elbow.elbow_plot(scores.ARGS.clustering_dir)
    elif ARGS.calc_optimal_num_clusters == 'gap':

    elif ARGS.num_clusters is not None:

        # add ARGS for number of clusters
        kmeans = KMeans(n_clusters=2, random_state=0)
        # gm = GaussianMixture(n_components=2, random_state=0)

        pred_kmeans = kmeans.fit_predict(bottleneck_outputs)
        # pred_gm = gm.fit_predict(bottleneck_outputs)

        log.info("predictions : {}".format(pred_kmeans))
        print("Cluster centers of Kmeans : ", kmeans.cluster_centers_)

        # log.info("predictions : {}".format(pred_gm))
        # print("Cluster centers of GaussianMixture : {:.8f}".format(gm.means_))

        df = pd.DataFrame(columns=["filename"] + ["cluster_number"])

        # print file names with respective cluster numbers
        for i in range(len(dataloader)):
            log.info("file name : {}, predicted cluster - Kmeans : {}".format(file_names[i], pred_kmeans[i]))
            # log.info("file name : {}, predicted cluster - GaussianMixture : {}".format(file_names[i], pred_gm[i]))

            df = df.append(dict(zip(df.columns, [file_names[i]] + [pred_kmeans[i]])), ignore_index=True)

        summary_dir = ARGS.summary_dir
        if summary_dir is not None:
            df.to_csv(summary_dir + "/Kmeans_clusters")








# -------------------------- leftover from Lucidmonkey --------------------------
# extract features
#df = pd.DataFrame( columns = ["filename"] + ["bottleneck" + str(_) for _ in range(nbottleneck)] )
#for spectrogram in data:
    #imagename = spectrogram["name"]
    #spectrogram = spectrogram["spectrogram"]

    # if the spectrogram is not of width 194 units, don't run the iteration
    #if spectrogram.shape[1] != 194:
        #continue

    #norm = np.linalg.norm(spectrogram)
    #snippet = spectrogram / norm
    #snippet = torch.reshape( snippet, (-1,) ).to(device)

    #features = model.generatefeatures(snippet).detach().numpy()
    #features = [str(f.item()) for f in features]
    #df = df.append( dict( zip( df.columns, [imagename] + list(features) ) ), ignore_index=True )

#df.to_csv(folder + "/features")

# perform clustering
#clustering = GaussianMixture(n_components=7, random_state=0).fit(df[[ "bottleneck" + str(_) for _ in range(nbottleneck) ]])

#df["clusterlabels"] = clustering.means_

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#r = [-1,1]
#X, Y = np.meshgrid(r, r)
#ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.savefig(folder + '/gmmmixtures')

# write to file
#df.to_csv(folder + "/features")

#sys.stdout.close()
#sys.stderr.close()
