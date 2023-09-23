
import torch
from byol_loader import BYOL, TwoCropsTransform, GaussianBlur
from torchvision import models
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import warnings
import torchvision.datasets as datasets
import pickle
warnings.filterwarnings("ignore")

from configs import *

device = 'cuda'

resnet = models.resnet50(pretrained=True).to(device)
resnet.state_dict(torch.load(os.path.join(root_dir, 'weights', dataset_name, 'resnet.pt')))

with open(os.path.join(root_dir, 'weights', dataset_name , 'paths'), "rb") as fp:   
   names = pickle.load(fp)

train_dataset = datasets.ImageFolder(image_dir, transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle = False, pin_memory=True, drop_last=True)

encoder = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool',
    return_embedding = True
)

embeddings = []

for j, (images, _) in enumerate(train_loader):
    with torch.no_grad():
        encoder.eval()
        images.to(device)
        embedding = encoder(image_one=images, image_two=None)
        embeddings.append(embedding[0].detach())

embeddings = torch.stack(embeddings)

# torch.save(embeddings, os.path.join(root_dir, 'weights', dataset_name, 'embeddings.pt'))

torch.save(embeddings, os.path.join(root_dir, 'weights', dataset_name, 'bg_embeddings.pt'))
