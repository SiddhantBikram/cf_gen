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
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

from configs import *

device = 'cuda'
mnist_dir = 'D:/Research/Counterfactual/Scripts/colored_mnist/train'
# mnist_dir = '/content/drive/MyDrive/Scripts/colored_mnist/train'

# resnet = models.resnet50(pretrained=True).to(device)

# learner = BYOL(
#     resnet,
#     image_size = 28,
#     hidden_layer = 'avgpool',
#     return_embedding = False
# )

# opt = torch.optim.Adam(learner.parameters(), lr=1e-5)

# augmentation = [
#             transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#         ]

# train_dataset = datasets.ImageFolder(mnist_dir, TwoCropsTransform(transforms.Compose(augmentation)))
# # print(len(train_dataset))

# # for i in range(9976): 
# #   print(train_dataset[i][1])

# # sub = list(range(0, len(train_dataset), 6))
# # train_dataset = torch.utils.data.Subset(train_dataset, sub)
# # print(len(train_dataset))
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle = False, pin_memory=True, drop_last=True)

# epochs = 1

# learner.train()

# print('Learning')

# for i in range(epochs):
#     for j, (images, _) in enumerate(train_loader):
#         print(i, j)

#         images[0] = images[0].cuda(non_blocking=True)
#         images[1] = images[1].cuda(non_blocking=True)

#         loss = learner(image_one=images[0], image_two=images[1])
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         learner.update_moving_average() 
#     print("Epoch: ",i," Loss: ", loss.item())

# torch.save(resnet.state_dict(),  os.path.join(root_dir, 'weights', 'mnist_model.pt'))

# resnet = models.resnet50(pretrained=True).to(device)
# resnet.state_dict(torch.load(os.path.join(root_dir, 'weights', 'mnist_model.pt')))

# learner = BYOL(
#     resnet,
#     image_size = 28,
#     hidden_layer = 'avgpool',
#     return_embedding = True
# )

# image_dataset = datasets.ImageFolder(mnist_dir, transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle = False, pin_memory=True, drop_last=True)
# print(len(train_loader))

# embeddings = []
# labels = []

# for j, (images, label) in enumerate(train_loader):
#     with torch.no_grad():
#         print(j)
#         learner.eval()
#         images.to(device)
#         embedding = learner(image_one=images, image_two=None)
#         embeddings.extend(embedding[0].detach())
#         labels.extend(label)

# embeddings = torch.stack(embeddings).cpu()
# labels = torch.stack(labels).cpu()

# torch.save(embeddings, os.path.join(root_dir, 'weights', 'mnist_embeddings.pt'))
# torch.save(labels, os.path.join(root_dir, 'weights', 'mnist_labels.pt'))

embeddings = torch.load(os.path.join(root_dir, 'weights', 'mnist_embeddings.pt'))
labels = torch.load( os.path.join(root_dir, 'weights', 'mnist_labels.pt'))

# new_embeddings = [] 
# new_labels = []

# for i in range(0, len(embeddings), 50):
#     new_embeddings.append(embeddings[i])
#     new_labels.append(labels[i])

# new_embeddings = torch.stack(new_embeddings).cpu()
# new_labels = torch.stack(new_labels).cpu()

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(embeddings)
df = pd.DataFrame()
df["y"] = labels
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]


sns.scatterplot(x="comp-1", y="comp-2", hue=labels.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title="MNIST data T-SNE projection")
plt.savefig(os.path.join(root_dir, 'scatterplot.png'))
plt.show()
