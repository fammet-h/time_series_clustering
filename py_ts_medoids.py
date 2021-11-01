"""
reference: https://y-uti.hatenablog.jp/entry/2016/01/07/154258
"""
import collections
import numpy as np
import pandas as pd
import random
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.spatial.distance import euclidean

class TimeSeriesKMedoids:

    def __init__(self, n_clusters=2, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def get_dtw_distance_matrix(self, list_of_dataframes):
        def dtw(coords1, coords2):
            """Dynamic Time Warping"""
            distance_matrix = [[float('inf') for _ in range(len(coords2) + 1)] for _ in range(len(coords1) + 1)]
            distance_matrix[0][0] = 0
            for i in range(1, len(coords1) + 1):
                for j in range(1, len(coords2) + 1):
                    distance_matrix[i][j] = min(
                        distance_matrix[i - 1][j - 1],
                        distance_matrix[i][j - 1],
                        distance_matrix[i - 1][j]
                    ) + euclidean(coords1[i - 1], coords2[j - 1])
            return distance_matrix[len(coords1)][len(coords2)]
        dtw_distance_matrix = [[None for _ in range(len(list_of_dataframes))] for _ in range(len(list_of_dataframes))]
        for i, p1 in enumerate(list_of_dataframes):
            for j, p2 in enumerate(list_of_dataframes):
                dtw_ij = dtw(p1.values.tolist(), p2.values.tolist())
                dtw_distance_matrix[i][j] = dtw_ij
        return dtw_distance_matrix

    def fit(self, dtw_distance_matrix):
        def _initialize_medoids(distances, n_clusters):
            medoids = [i for i in range(len(distances))]
            random.shuffle(medoids)
            return medoids[0:n_clusters]
        def _making_initial_medoids(distances, n_clusters):
            """
            making initial medoids
            """
            n = len(distances)
            distances_pd = pd.DataFrame({'id':range(n)})
            distances_pd = pd.concat([distances_pd, pd.DataFrame(distances, columns=[i for i in range(n)])], axis=1)
            medoids = []
            for cluster_num in range(n_clusters):
                if cluster_num == 0:
                    medoid = np.random.randint(0, n, size=1)
                    medoids.extend(medoid)
                else:
                    distance = distances_pd.drop(medoids, axis=0)
                    distance = distance.loc[:, ['id'] + medoids]
                    distance['min_distance'] = distance.min(axis=1)
                    distance['min_distance_squared'] = distance['min_distance'] ** 2
                    ids = distance['id'].values
                    distance_values = distance['min_distance_squared'] / np.sum(distance['min_distance_squared'])
                    medoid = ids[np.random.choice(range(ids.size), 1, p=distance_values)]
                    medoids.extend(medoid)
            medoids = sorted(medoids)
            return medoids
        def _assign_to_nearest(distances, medoids):
            n_clusters = len(medoids)
            indices = []
            for distance in distances:
                min_dist = float('inf')
                nearest = 0
                for i in range(n_clusters):
                    if distance[medoids[i]] < min_dist:
                        min_dist = distance[medoids[i]]
                        nearest = i
                indices.append(nearest)
            return indices
        def _update_medoids(distances, indices, n_clusters):
            len_distances = len(distances)
            min_dists = [float('inf') for _ in range(n_clusters)]
            medoids = [False for _ in range(n_clusters)]
            for i in range(len_distances):
                medoids_index = indices[i]
                dist = 0
                for j in range(len_distances):
                    if indices[j] == medoids_index:
                        dist += distances[i][j]
                if dist < min_dists[medoids_index]:
                    min_dists[medoids_index] = dist
                    medoids[medoids_index] = i
            return medoids
        medoids = _making_initial_medoids(dtw_distance_matrix, self.n_clusters)
        indices = False
        i = 0
        while (i < self.max_iter):
            next = _assign_to_nearest(dtw_distance_matrix, medoids)
            if next == indices:
                break
            indices = next
            medoids = _update_medoids(dtw_distance_matrix, indices, self.n_clusters)
            i += 1
        return indices, medoids


class TimeSeriesGMedoids(TimeSeriesKMedoids):
    def __init__(self, n_init_clusters=2, max_iter=10):
        self.n_init_clusters = n_init_clusters
        self.max_iter = max_iter

    def fit(self, list_of_dataframes):
        n_clusters = self.n_init_clusters
        tskm = TimeSeriesKMedoids(n_clusters, self.max_iter)
        dtw_distance_matrix = tskm.get_dtw_distance_matrix(list_of_dataframes)
        distances_ij = []
        for i in range(n_clusters):
            for j in range(i, n_clusters):
                distances_ij.append(dtw_distance_matrix[i][j])
        _, p_value = shapiro(np.array(distances_ij))
        while True:
            tskm = TimeSeriesKMedoids(n_clusters, self.max_iter)
            indices, medoids = tskm.fit(dtw_distance_matrix)
            counter = collections.Counter(indices)
            min_cluster_counts = counter.most_common()[-1][1]
            if min_cluster_counts <= 3:
                break
            p_values = []
            for i in range(n_clusters):
                _, p_value_i = shapiro(np.array([
                    dtw_distance_matrix[index][medoids[i]]
                    for index, label
                    in enumerate(indices)
                    if label == i]))
                p_values.append(p_value_i)
            if max(p_values) < p_value:
                break
            p_value = max(p_values)
            n_clusters += 1
        return indices, medoids
