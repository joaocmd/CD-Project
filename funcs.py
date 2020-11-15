import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def cluster_Kmeans(data, v1, v2, N_CLUSTERS, estimatorType):
    rows, cols = ds.choose_grid(len(N_CLUSTERS))
    pbar = tqdm(total=(len(N_CLUSTERS)))

    mse: list = []
    sc: list = []
    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)): 
        pbar.update(1)
        if (estimatorType == 'Kmeans'):
            k = N_CLUSTERS[n]
            estimator = KMeans(n_clusters=k)
            estimator.fit(data)
            mse.append(estimator.inertia_)
            sc.append(silhouette_score(data, estimator.labels_))
            ds.plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k,
                         f'KMeans k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
            
            
        elif (estimatorType == 'Gaussian'):
            k = N_CLUSTERS[n]
            estimator = GaussianMixture(n_components=k)
            estimator.fit(data)
            labels = estimator.predict(data)
            mse.append(ds.compute_mse(data.values, labels, estimator.means_))
            sc.append(silhouette_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.means_, k,
                     f'EM k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
            
            
        else:
            estimator = DBSCAN(eps=N_CLUSTERS[n], min_samples=2)
            estimator.fit(data)
            labels = estimator.labels_
            k = len(set(labels)) - (1 if -1 in labels else 0)
            if k > 1:
                centers = ds.compute_centroids(data, labels)
                mse.append(ds.compute_mse(data.values, labels, centers))
                sc.append(silhouette_score(data, labels))
                ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k,
                                 f'DBSCAN eps={N_CLUSTERS[n]} k={k}', ax=axs[i,j])
                i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
            else:
                mse.append(0)
                sc.append(0)
                
    pbar.close()
    plt.show()
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    
    if (estimatorType == 'Kmeans'):
        ds.plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
        
    elif (estimatorType == 'Gaussian'):
        ds.plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
        
    else:
        ds.plot_line(N_CLUSTERS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 1], percentage=True)
        
    plt.show()
    
def densityBasedCluster(data, v1, v2, EPS):
    mse: list = []
    sc: list = []
    rows, cols = ds.choose_grid(len(EPS))
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    pbar = tqdm(total=(len(EPS)))
    for n in range(len(EPS)):
        estimator = DBSCAN(eps=EPS[n], min_samples=2)
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = ds.compute_centroids(data, labels)
            mse.append(ds.compute_mse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k,
                             f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        else:
            mse.append(0)
            sc.append(0)
        pbar.update(1)
    pbar.close()
    plt.show()

def metric(data_received, v1, v2):
    data = data_received.copy()
    METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
    distances = []
    for m in METRICS:
        dist = np.mean(np.mean(squareform(pdist(data.values, metric=m))))
        distances.append(dist)

    print('AVG distances among records', distances)
    distances[0] *= 0.6
    distances[1] = 80
    distances[2] *= 0.6
    distances[3] *= 0.1
    distances[4] *= 0.15
    print('CHOSEN EPS', distances)
    
    mse: list = []
    sc: list = []
    rows, cols = ds.choose_grid(len(METRICS))
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(METRICS)):
        estimator = DBSCAN(eps=distances[n], min_samples=2, metric=METRICS[n])
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = ds.compute_centroids(data, labels)
            mse.append(ds.compute_mse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k,
                             f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i,j])
        else:
            mse.append(0)
            sc.append(0)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.show()
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    ds.bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
    plt.show()
    
def hierarchical(data_received, v1, v2):
    data = data_received.copy()

    N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

    mse: list = []
    sc: list = []
    rows, cols = ds.choose_grid(len(N_CLUSTERS))
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = AgglomerativeClustering(n_clusters=k)
        estimator.fit(data)
        labels = estimator.labels_
        centers = ds.compute_centroids(data, labels)
        mse.append(ds.compute_mse(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
        ds.plot_clusters(data, v2, v1, labels, centers, k,
                         f'Hierarchical k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
    plt.show()

    METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
    LINKS = ['complete', 'average']
    k = 3
    values_mse = {}
    values_sc = {}
    rows = len(METRICS)
    cols = len(LINKS)
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    for i in range(len(METRICS)):
        mse: list = []
        sc: list = []
        m = METRICS[i]
        for j in range(len(LINKS)):
            link = LINKS[j]
            estimator = AgglomerativeClustering(n_clusters=k, linkage=link, affinity=m )
            estimator.fit(data)
            labels = estimator.labels_
            centers = ds.compute_centroids(data, labels)
            mse.append(ds.compute_mse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels, centers, k,
                             f'Hierarchical k={k} metric={m} link={link}', ax=axs[i,j])
        values_mse[m] = mse
        values_sc[m] = sc
    plt.show()

    _, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    ds.multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
    plt.show()