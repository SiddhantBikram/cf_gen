import io
import torch
import os
from configs import *

with open(os.path.join(root_dir, 'weights', dataset_name, 'embeddings.pt'), 'rb') as f:
    embeddings = io.BytesIO(f.read())