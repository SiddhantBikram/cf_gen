import enum
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset 
import pickle
import time 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve
import random 
import torch.backends.cudnn as cudnn
import json
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import os
import time
import numpy as np
import faiss
import torch
import sys
import torch.nn.functional as F
from torch.nn import Parameter
import math
from configs import *

def preprocess_features(npdata, pca):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """

    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception("nan occurs")
    if pca != -1:
        print("\nPCA from dim {} to dim {}".format(ndim, pca))
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception(
                "More than 0.1% nan occurs after pca, percent: {}%".format(
                    percent))
        else:
            npdata[np.isnan(npdata)] = 0.
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata

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
    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.useFloat16 = False
    # flat_config.device = 0
    index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]

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

def reduce(feats, k):
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

    # os.makedirs(f'datasets_deconf/{args.dataset}', exist_ok=True)
    prototypes.append(centroids)
    prototypes = np.array(prototypes)
    prototypes =  prototypes.reshape(-1, image_dim)
    # print(f'datasets_deconf/{args.dataset}/train_bag_cls_agnostic_feats_proto_{k}.npy')
    # np.save(f'datasets_deconf/{args.dataset}/train_bag_cls_agnostic_feats_proto_{k}.npy', prototypes)

    del feats
    return prototypes

class CCIM(nn.Module):
  def __init__(self, num_joint_feature, num_gz, strategy):
    super(CCIM, self).__init__()
    self.num_joint_feature = num_joint_feature
    self.num_gz = num_gz
    if strategy == 'dp_cause':
      self.causal_intervention = dot_product_intervention(num_gz, num_joint_feature )
    elif strategy == 'ad_cause':
      self.causal_intervention = additive_intervention(num_gz, num_joint_feature )
    else:
      raise ValueError("Do Not Exist This Strategy.")

    self.w_h = Parameter(torch.Tensor(self.num_joint_feature, 128)) 
    self.w_g = Parameter(torch.Tensor(self.num_gz, 128)) 
    self.final_fc = nn.Linear(128, 1)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_normal_(self.w_h)
    nn.init.xavier_normal_(self.w_g)

  def forward(self, joint_feature, confounder_dictionary):   
    g_z = self.causal_intervention(confounder_dictionary, joint_feature)
    proj_h = torch.matmul(joint_feature, self.w_h)  
    proj_g_z = torch.matmul(g_z, self.w_g) 
    do_x = proj_h + proj_g_z
    do_x = self.final_fc(do_x)

    return do_x


def gelu(x):
      return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class dot_product_intervention(nn.Module):
  def __init__(self, con_size, fuse_size):
    super(dot_product_intervention, self).__init__()
    self.con_size = con_size 
    self.fuse_size = fuse_size  
    self.query = nn.Linear(self.fuse_size, 256, bias= False) 
    self.key = nn.Linear(self.con_size, 256, bias= False) 


  def forward(self, confounder_set, fuse_rep):
    query = self.query(fuse_rep) 
    key = self.key(confounder_set)  
    mid = torch.matmul(query, key.transpose(0,1)) / math.sqrt(self.con_size)  
    attention = F.softmax(mid, dim=-1) 
    attention = attention.unsqueeze(2)  
    fin = (attention*confounder_set).sum(1)  
    
    return fin

class additive_intervention(nn.Module):
  def __init__(self, con_size, fuse_size):
    super(additive_intervention,self).__init__()
    self.con_size = con_size
    self.fuse_size = fuse_size
    self.Tan = nn.Tanh()
    self.query = nn.Linear(self.fuse_size, 256, bias = False)
    self.key = nn.Linear(self.con_size, 256, bias = False)
    self.w_t = nn.Linear(256, 1, bias=False)

  def forward(self, confounder_set, fuse_rep):
 
    query = self.query(fuse_rep) 
  
    key =  self.key(confounder_set)  
  
    query_expand = query.unsqueeze(1)  
    fuse = query_expand + key 
    fuse = self.Tan(fuse)
    attention = self.w_t(fuse) 
    attention = F.softmax(attention, dim=1)
    fin = (attention*confounder_set).sum(1)  

    return fin