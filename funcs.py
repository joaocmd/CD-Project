import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
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