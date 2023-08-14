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

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool',
    return_embedding = False
)

opt = torch.optim.Adam(learner.parameters(), lr=1e-5)

augmentation = [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]

train_dataset = datasets.ImageFolder(image_dir, TwoCropsTransform(transforms.Compose(augmentation)))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle = False, pin_memory=True, drop_last=True)

print('Images loaded.')

epochs = 4

learner.train()

for i in range(epochs):
    for j, (images, _) in enumerate(train_loader):

        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        loss = learner(image_one=images[0], image_two=images[1])
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() 
    print("Epoch: ",i," Loss: ", loss.item())

torch.save(resnet.state_dict(),  os.path.join(root_dir, 'weights', dataset_name, 'model.pt'))

