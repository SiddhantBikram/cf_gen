#CCIM and IBMIL

import io
import torch
import torch.nn.functional as F
import os
from configs import *
import pickle
import numpy as np
from backdoor_loader import *
import psutil
import sys

n_clusters = 9

bg_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'bg_embeddings.pt'))
obj_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'embeddings.pt'))
bg_embeddings = bg_embeddings.squeeze(1)
obj_embeddings = obj_embeddings.squeeze(1)

conf_dict = reduce(bg_embeddings, n_clusters)
conf_dict = torch.from_numpy(conf_dict)

torch.save(conf_dict, os.path.join(root_dir, 'weights', dataset_name, 'conf_dict.pt'))

lister = []

for i in range(len(bg_embeddings)):
    print('RAM memory % used:', psutil.virtual_memory()[2])

    joint_rep = torch.cat((obj_embeddings[i], bg_embeddings[i]), 0).unsqueeze(1).cpu()
    ccim = CCIM(1, image_dim, strategy='dp_cause')
    out = ccim(joint_rep, conf_dict)
    # print(sys.getsizeof(out))
    # print(sys.getsizeof(joint_rep))
    lister.append(out.detach())
