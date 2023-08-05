import torch
from byol_loader import BYOL
from torchvision import models
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

device = 'cuda'

resnet = models.resnet50(pretrained=True).to(device)

learner = BYOL(
    resnet,
    image_size = 32,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

def mnist():
    path = "C:\\Users\\siddh\\Desktop\\Counterfactual\\MNIST-M\\MNIST-M\\training\\sample"
    imgs = []
    convert_tensor = transforms.ToTensor()

    print('Loading images')

    for img in os.listdir(path):
        image = Image.open(path+'/'+img)
        # image = image.resize((32,32))
        image = convert_tensor(image)
        imgs.append(image)

    imgs = torch.stack(imgs)
    return imgs

images = mnist().to(device)

print('Done!')

for i in range(10):
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder
    print("Epoch: ",i," Loss: ", loss.item())

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')