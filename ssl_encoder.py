import torch
from byol_loader import BYOL, TwoCropsTransform
from torchvision import models
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import warnings
import torchvision.datasets as datasets

warnings.filterwarnings("ignore")

device = 'cuda'

root_dir = 'D:/Research/Counterfactual/Scripts/'
image_dir = os.path.join(root_dir, 'IN9sub')

resnet = models.resnet50(pretrained=True).to(device)

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool',
    return_embedding = False
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

augmentation = [
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([byol.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

train_dataset = datasets.ImageFolder(root = image_dir, TwoCropsTransform(transforms.Compose(augmentation)))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16,
    num_workers=args.workers, shuffle = False, pin_memory=True, drop_last=True)

print('Images loaded.')

epochs = 20

for i in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        learner.train()
        loss = learner(images)
        print(loss[0].shape)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        print("Epoch: ",i," Loss: ", loss.item())

torch.save(resnet.state_dict(),  os.path.join(root_dir, 'weights', 'BYOL_weights.pt'))

def IM9():
    imgs = []
    names = []
    convert_tensor = transforms.ToTensor()

    print('Loading images')

    for root, dirs, files in os.walk(image_dir):
	    for img_name in files:
                image = Image.open(os.path.join(root, img_name))
                image = convert_tensor(image)
                imgs.append(image)
                names.append(img_name)

    imgs = torch.stack(imgs)
    return imgs, names

images, names = IM9()
images = images.to(device)

encoder = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool',
    return_embedding = True
)

model.eval()
