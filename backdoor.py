#CCIM and IBMIL

import io
import torch
import torch.nn.functional as F
import os
from configs import *
import pickle
import numpy as np

embeddings = torch.load(os.path.join(root_dir, 'weights', dataset_name, 'bg_embeddings.pt'))

print(embeddings.shape)


