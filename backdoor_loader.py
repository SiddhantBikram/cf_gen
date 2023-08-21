from turtle import forward
from numpy import rint
from sklearn import datasets
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math
from torch.nn import Parameter

class Kmeans:

    def __init__(self, k, pca_dim=256):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat, verbose=False, seed=None):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(feat, self.pca_dim)

        # cluster the data
        I = run_kmeans(xb, self.k, verbose, seed)
        self.labels = np.array(I)
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

def reduce(args, feats, k):
    '''
    feats:bag feature tensor,[N,D]
    k: number of clusters
    shift: number of cov interpolation
    '''
    prototypes = []
    semantic_shifts = []
    feats = feats.cpu().numpy()

    kmeans = Kmeans(k=k, pca_dim=-1)
    kmeans.cluster(feats, seed=66)  # for reproducibility
    assignments = kmeans.labels.astype(np.int64)
    # compute the centroids for each cluster
    centroids = np.array([np.mean(feats[assignments == i], axis=0)
                          for i in range(k)])

    # compute covariance matrix for each cluster
    covariance = np.array([np.cov(feats[assignments == i].T)
                           for i in range(k)])

    os.makedirs(f'datasets_deconf/{args.dataset}', exist_ok=True)
    prototypes.append(centroids)
    prototypes = np.array(prototypes)
    prototypes =  prototypes.reshape(-1, args.feats_size)
    print(prototypes.shape)
    print(f'datasets_deconf/{args.dataset}/train_bag_cls_agnostic_feats_proto_{k}.npy')
    np.save(f'datasets_deconf/{args.dataset}/train_bag_cls_agnostic_feats_proto_{k}.npy', prototypes)

    del feats

def run_kmeans(x, nmb_clusters, verbose=False, seed=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    if seed is not None:
        clus.seed = seed
    else:
        clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]