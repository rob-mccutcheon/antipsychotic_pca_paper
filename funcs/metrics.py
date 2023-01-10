import numpy as np
import bct as bc

def get_clusters(ki_corr, gamma):
    Q = []
    for i in range(100):
        np.random.seed(i)
        Q.append(bc.community_louvain(ki_corr, B='negative_sym', gamma=gamma)[1])
    seed = np.argmax(Q)
    np.random.seed(seed)
    clusters =  bc.community_louvain(ki_corr, B='negative_sym', gamma=gamma)[0]
    clusters_sorted = np.array(sorted(clusters))
    shifts = np.where(clusters_sorted[:-1] != clusters_sorted[1:])[0]+1
    cluster_idxs = np.atleast_2d(np.array([]))
    for i in range(np.max(clusters)):
        cluster_idxs = np.hstack([cluster_idxs, np.where(clusters==i+1)])
    cluster_idxs = list(cluster_idxs.flatten().astype(int))
    return cluster_idxs, shifts