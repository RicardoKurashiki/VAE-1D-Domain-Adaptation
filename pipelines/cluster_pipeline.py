import os
import numpy as np
from sklearn.cluster import KMeans


def run(model_path, dataset_name, k, source_dataset_name=None, features_dir=None):
    """Fita KMeans no source e salva os centróides (protótipos por classe) usados
    como âncoras da CenterLoss no treinamento do autoencoder de alinhamento.
    Datasets alvo não geram centróides próprios — não são plotados nem usados."""
    if source_dataset_name is None:
        source_dataset_name = dataset_name

    if dataset_name != source_dataset_name:
        return

    if features_dir is None:
        features_dir = os.path.join(model_path, "features")

    centroids_dir = os.path.join(model_path, "centroids")
    os.makedirs(centroids_dir, exist_ok=True)

    train_features = np.load(os.path.join(features_dir, f"{dataset_name}_train_features.npy"))
    train_labels = np.load(os.path.join(features_dir, f"{dataset_name}_train_labels.npy"))

    n_clusters = k if k is not None else len(np.unique(train_labels))
    clusterer = KMeans(n_clusters=n_clusters, random_state=42).fit(train_features)

    np.save(os.path.join(centroids_dir, "cluster_centers.npy"), clusterer.cluster_centers_)

    centroid_labels = np.array([
        np.bincount(train_labels[clusterer.labels_ == c].astype(int)).argmax()
        for c in range(n_clusters)
    ])
    np.save(os.path.join(centroids_dir, "centroid_labels.npy"), centroid_labels)
