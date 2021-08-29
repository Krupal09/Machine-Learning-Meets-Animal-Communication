#!/usr/bin/env python3

"""
Module: clustering.py

Clustering for convolutional autoencoder :
* Iterate through the whole dataset(both training and validation) passing them
  through only encoder to get bottleneck vectors for each data point.
* Cluster these bottleneck vectors using unsupervised clustering methods like
  Kmeans or GaussianMixture modeling.

Online resources we find useful during implementation
Methods for selecting the optimal number of clusters: https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
Gap statistics intro: https://glowingpython.blogspot.com/2019/01/a-visual-introduction-to-gap-statistics.html
KMeans, elbow and Silhouette: https://realpython.com/k-means-clustering-python/

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
from kneed import KneeLocator # to identify the elbow point programmatically

parser = argparse.ArgumentParser()

""" Directory parameters """

parser.add_argument(
    "--clustering_dir",
    type=str,
    help="The directory where the clustering outputs will be stored.",
)

""" Clustering parameters """

parser.add_argument(
    "--calc_optimal_num_clusters", type=str, default=None, help="use elbow or gap statistics or both (comparison) to find the optimal number of clusters",
)

parser.add_argument(
    "--num_clusters", type=int, default=None, help="The number of clusters to cluster all the data points."
)

parser.add_argument(
    "--clustering_algorithm",
    type=str,
    default=None,
    help="Use either KMeans or GMM to cluster the embeddings",
)


ARGS = parser.parse_args()

kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

class Elbow:
    """ Finding optimal number of clusters, by plotting elbow curve on KMeans scores (negative of the K-means objective)"""

    def __init__(self, num_clusters=range(1,15), **kmeans_kwargs):
        self._num_clusters = num_clusters
        self._init = init
        self._random_state = random_state
        self._max_iter = max_iter

    def kmeans(self, inputs):
        kmeans = [KMeans(
            n_clusters=i,
            init = self._init,
            random_state = self._random_state,
            max_iter = self._max_iter,
        ) for i in self._num_clusters]
        scores = [kmeans[i].fit(inputs).inertia_ for i in range(len(kmeans))]
        return scores

    def elbow_plot(self, scores, path):
        plt.plot(self._num_clusters, scores)
        plt.xlabel('Number of clusters')
        plt.ylabel('Score (negative of the K-means objective)')
        plt.title('Elbow curve')
        plt.savefig(os.path.join(path, 'ElbowCurve.png'))

    def get_elbow(self, scores):
        kl = KneeLocator(
            self._num_clusters, scores, curve
        )
        return kl.elbow


def gap_optimalK(data, num_refs=3, maxClusters=15):
    """
    Calculate kMeans optimal number of clusters using Gap statistics

    :param data: (n_samples, n_features)
    :param num_refs: number of sample reference datasets to create
    :param maxClusters: Maximum number of clusters to test for

    :return: (gaps, optimalK)
    """

    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})

    for gap_index, k in enumerate(range(1, maxClusters)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp  # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_  # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)  # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)
        return (gaps.argmax() + 1, resultsdf)
        score_g, df = optimalK(cluster_df, nrefs=5, maxClusters=30)
        plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('Gap Statistic');
    plt.title('Gap Statistic vs. K');



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

    # calculate the optimal number of clusters using elbow or gap statistics
    if ARGS.calc_optimal_num_clusters is not None:

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

        elif ARGS.calc_optimal_num_clusters == 'comparison':



    # if you already know how many clusters you would like to have,
    # you could directly train the clustering model and cluster the embeddings
    if ARGS.num_clusters is not None:

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

        if ARGS.clustering_dir is not None:
            df.to_csv(ARGS.clustering_dir + "/Kmeans_clusters")








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
