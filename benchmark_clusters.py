import pandas as pd
import numpy as np
import pickle
from time import time
from sys import argv
from random import randint
from gmeans import gmeans
from pyclustering.cluster.gmeans import gmeans as pyc_gmeans
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation


def save_to_pickle(obj, file_path):
    with open(file_path, "wb") as fp:
        pickle.dump(obj, fp)


if __name__ == "__main__":

    # parse input sizes
    sizes = map(int, argv[1].split(","))

    # load dataset
    df = pd.read_pickle(argv[2])

    # make output file
    outputs = {
        "samples": {},
        "results": {}
    }
    output_path = argv[3]

    random_seed = randint(0, 100000)

    for size in sizes:
        all_c = []
        sample = df.sample(size, random_state=random_seed).values
        outputs["samples"][size] = sample
        outputs["results"][size] = {}
        save_to_pickle(outputs, output_path)

        # 1. run gmeans x 3
        outputs["results"][size]["gmeans"] = []
        for run in range(3):
            print("running g-means on " + str(size) + " samples...")
            start = time()
            _, labels, centers = gmeans(sample)
            elapsed = time() - start
            print("g-means finished in " + str(elapsed) +
                  "s, cl=" + str(len(centers)))
            result = {
                "labels": labels,
                "centers": centers,
                "elapsed": elapsed
            }
            outputs["results"][size]["gmeans"].append(result)
            all_c.append(len(centers))
            save_to_pickle(outputs, output_path)

        # 2. run pyclustering.gmeans x 3
        outputs["results"][size]["pyc_gmeans"] = []
        for run in range(3):
            print("running pyclustering g-means on " +
                  str(size) + " samples...")
            start = time()
            pygm = pyc_gmeans(sample)
            pygm.process()
            elapsed = time() - start
            labels = np.full((len(sample),), None)
            centers = np.array(pygm.get_centers())
            for ci, c in enumerate(pygm.get_clusters()):
                for i in c:
                    labels[i] = ci
            print("pyclustering g-means finished in " + str(elapsed) +
                  "s, cl=" + str(len(centers)))
            result = {
                "labels": labels,
                "centers": centers,
                "elapsed": elapsed
            }
            outputs["results"][size]["pyc_gmeans"].append(result)
            all_c.append(len(centers))
            save_to_pickle(outputs, output_path)

        # 3. run Ward clustering, with k = largest from k-means
        outputs["results"][size]["ward"] = []
        for run in range(1):
            print("running ward clustering on " +
                  str(size) + " samples... c=" + str(max(all_c)))
            start = time()
            wa = AgglomerativeClustering(
                n_clusters=max(all_c), linkage='ward').fit(sample)
            elapsed = time() - start
            print("ward finished in " + str(elapsed) +
                  "s, cl=" + str(max(all_c)))
            result = {
                "labels": wa.labels_,
                "centers": max(all_c),
                "elapsed": elapsed
            }
            outputs["results"][size]["ward"].append(result)
            save_to_pickle(outputs, output_path)
