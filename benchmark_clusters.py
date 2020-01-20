import pandas as pd
import numpy as np
import pickle
from time import time
from sys import argv
from random import randint
from gmeans import gmeans
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.metrics import pairwise_distances
from memory_profiler import profile


def save_to_pickle(obj, file_path):
    with open(file_path, "wb") as fp:
        pickle.dump(obj, fp)


@profile
def run_gmeans(sample):
    start = time()
    _, labels, centers = gmeans(sample)
    elapsed = time() - start
    print("g-means finished in " + str(elapsed) +
          "s, cl=" + str(len(centers)))
    result = {
        "labels": labels,
        "centers": np.uint8(centers),
        "elapsed": elapsed
    }
    return result


@profile
def run_ward(sample, cl_numbers):
    start = time()
    wa = AgglomerativeClustering(
        n_clusters=cl_numbers, linkage='ward').fit(sample)
    elapsed = time() - start
    print("ward finished in " + str(elapsed) +
          "s, cl=" + str(cl_numbers))
    cls = [[] for i in range(cl_numbers)]
    for i, c in enumerate(wa.labels_):
        cls[c].append(i)
    result = {
        "labels": wa.labels_,
        "centers": np.uint8(np.array([
            pd.DataFrame(sample[members]).mean().values for members in cls])),
        "elapsed": elapsed
    }
    return result


@profile
def run_kmeans(sample, cl_numbers):
    start = time()
    km = KMeans(
        n_clusters=cl_numbers, init="k-means++",
        tol=0.025, n_jobs=-1, n_init=1).fit(sample)
    elapsed = time() - start
    print("k-means finished in " + str(elapsed) +
          "s, cl=" + str(cl_numbers))
    result = {
        "labels": km.labels_,
        "centers": np.uint8(km.cluster_centers_),
        "elapsed": elapsed
    }
    return result


@profile
def run_birch(sample, threshold):
    start = time()
    bc = Birch(threshold=threshold,
               branching_factor=len(sample),
               n_clusters=None).fit(sample.astype(np.float))
    elapsed = time() - start
    result = {
        "labels": bc.labels_,
        "centers": np.uint8(bc.subcluster_centers_),
        "elapsed": elapsed
    }
    print("birch finished in " + str(elapsed) +
          "s, cl=" + str(len(result["centers"])))
    return result


if __name__ == "__main__":

    # parse input sizes
    sizes = map(int, argv[1].split(","))

    # parse how many times gmeans, ward, and kmeans
    # is to ran
    gmeans_count, ward_count, kmeans_count, birch_count = map(
        int, argv[2].split(","))

    # load dataset
    df = pd.read_pickle(argv[3])

    # make output file
    outputs = {
        "samples": {},
        "samples_idx": {},
        "colnames": df.colnames,
        "input_filename": argv[3],
        "results": {}
    }
    output_path = argv[4]

    if len(argv) > 5:
        random_seed = int(argv[5])
    else:
        random_seed = randint(0, 100000)

    for size in sizes:
        df_sample = df.sample(size, random_state=random_seed)
        sample = df_sample.values
        outputs["random_seed"] = random_seed
        outputs["samples"][size] = sample
        outputs["samples_idx"][size] = df_sample.index
        outputs["results"][size] = {}
        save_to_pickle(outputs, output_path)

        # 1. run gmeans
        outputs["results"][size]["gmeans"] = []
        for run in range(gmeans_count):
            print("running g-means on " + str(size) + " samples...")
            result = run_gmeans(sample)
            outputs["results"][size]["gmeans"].append(result)
            save_to_pickle(outputs, output_path)

        # 2. run Ward clustering, with k = 0.4*n
        outputs["results"][size]["ward"] = []
        for run in range(ward_count):
            cl_numbers = int(0.4 * size)
            print("running ward clustering on " +
                  str(size) + " samples... c=" + str(cl_numbers))
            result = run_ward(sample, cl_numbers)
            outputs["results"][size]["ward"].append(result)
            save_to_pickle(outputs, output_path)

        # 3. run k-means clustering, with k = 0.4*n
        outputs["results"][size]["kmeans"] = []
        for run in range(kmeans_count):
            cl_numbers = int(0.4 * size)
            print("running k-means on " +
                  str(size) + " samples... c=" + str(cl_numbers))
            result = run_kmeans(sample, cl_numbers)
            outputs["results"][size]["kmeans"].append(result)
            save_to_pickle(outputs, output_path)

        # 3. run birch
        outputs["results"][size]["birch"] = []
        for run in range(birch_count):
            shuffled_rand = randint(0, 100000)
            shuffled_df = df_sample.sample(
                len(df_sample), random_state=random_seed)
            print("running birch on " +
                  str(size) + " samples...")
            threshold = np.array([np.percentile(
                pairwise_distances(df_sample.sample(
                    min(
                        1000, int(
                            0.1 * len(
                                df_sample)))).values, metric='euclidean'
                ), 1) for i in range(100)]
            ).mean()
            result = run_birch(shuffled_df.values, threshold)
            result["random_seed"] = shuffled_rand
            result["samples_idx"] = shuffled_df.index
            outputs["results"][size]["birch"].append(result)
            save_to_pickle(outputs, output_path)
