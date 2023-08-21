import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import warnings

import pickle
warnings.filterwarnings("ignore")

from configs import *

def imfolder(dir):
    imgs = []
    names = []
    labels = []
    convert_tensor = transforms.ToTensor()

    print('Loading images')

    for root, dirs, files in os.walk(dir):
	    for img_name in files:
                image = Image.open(os.path.join(root,  img_name))
                image = image.resize((256,256))
                image = convert_tensor(image)
                imgs.append(image)
                names.append(os.path.join(root, img_name))
                labels.append(root.split('\\')[-1])

    imgs = torch.stack(imgs)
    return imgs, names, labels

images, names, labels = imfolder(image_dir)

torch.save(images, os.path.join(root_dir, 'weights', dataset_name, 'features.pt'))

# torch.save(images, os.path.join(root_dir, 'weights', dataset_name, '_bg_features.pt'))

# with open(os.path.join(root_dir, 'weights', dataset_name + 'paths'), 'wb') as fp:
#     pickle.dump(names, fp)

# with open(os.path.join(root_dir, 'weights', dataset_name, 'labels'), 'wb') as f:
#     pickle.dump(labels, f)