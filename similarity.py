import io
import torch
import torch.nn.functional as F
import os
from configs import *
import pickle
import numpy as np

embeddings = torch.load(os.path.join(root_dir, 'weights', dataset_name, 'embeddings.pt'))

similar = []

for i in range(20):
    max = 0
    cross_value = []
    for j in range(embeddings.shape[0]):
        if i ==j:
            continue

        sim = F.cosine_similarity(embeddings[i], embeddings[j])
        cross_value.append(sim[0].item())

    ind = np.argpartition(np.array(cross_value), -5)[-5:]
    similar.append(ind)

with open(os.path.join(root_dir, 'weights', dataset_name , 'similar_5.pkl'), 'wb') as fp:
    pickle.dump(similar, fp)
