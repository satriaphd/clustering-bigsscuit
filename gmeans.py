import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import anderson
import warnings


def get_clusters(km):
    "get clusters array from fitted k-means object"
    return np.array([np.where(km.labels_ == i)
                     for i in range(len(km.cluster_centers_))])


def expand_clusters(data, clusters, centers):
    "for every cluster, try split into two"
    new_centers = []
    for i, cluster in enumerate(clusters):
        if len(centers) == 1:
            data_cluster = data
        else:
            data_cluster = data[tuple(cluster)]

        if len(data_cluster) > 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=2, init="k-means++", n_init=1,
                            precompute_distances=True,
                            tol=0.025,
                            algorithm="full").fit(data_cluster)
                if (len(set(km.labels_)) == 2 and
                        not accept_test(data_cluster, km.cluster_centers_)):
                    new_centers.extend(km.cluster_centers_)
                    continue
        new_centers.append(centers[i])

    return np.array(new_centers)


def accept_test(data, centers):
    "perform Anderson-Darling test"
    assert(len(centers) == 2)
    v = np.subtract(centers[0], centers[1])
    square_norm = np.sum(np.multiply(v, v))
    points = np.divide(np.sum(np.multiply(data, v), axis=1), square_norm)

    estimation, critical, _ = anderson(points, dist='norm')
    return estimation < critical[-1]


def gmeans(data):
    # perform initial clustering with k=1
    km = KMeans(n_clusters=1, init="random", n_init=1,
                tol=0.025,
                precompute_distances=True, algorithm="full").fit(data)
    clusters, labels, centers = get_clusters(
        km), km.labels_, km.cluster_centers_
    while True:
        new_centers = expand_clusters(data, clusters, centers)
        if len(centers) == len(new_centers):  # convergence
            break
        # re-perform k-means, with existing centers
        km = KMeans(n_clusters=len(new_centers), init=new_centers, n_init=1,
                    precompute_distances=True,
                    tol=0.025,
                    algorithm="full").fit(data)
        clusters, labels, centers = get_clusters(
            km), km.labels_, km.cluster_centers_

    return clusters, labels, centers
