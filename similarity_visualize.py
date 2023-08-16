import io
import torch
import torch.nn.functional as F
import os
from configs import *
import pickle
import cv2
from matplotlib import pyplot as plt
from PIL import Image

device = 'cuda'

with open(os.path.join(root_dir, 'weights', dataset_name, 'similar_5.pkl'), 'rb') as fp:
    similar = pickle.load(fp)

with open(os.path.join(root_dir, 'weights', dataset_name , 'paths'), "rb") as fp:   
    names = pickle.load(fp)

print(len(similar))

for i in  range(20):
    fig = plt.figure(figsize=(10, 7))

    fig.add_subplot(2, 2, 1)
  
    plt.imshow(Image.open(os.path.join(object_dir, names[i][len(image_dir)+1:].split('.')[0] + '.png')))
    plt.axis('off')
    plt.title("First")

    # Adds a subplot at the 2nd position
    fig.add_subplot(2, 2, 2)

    # showing image
    plt.imshow(Image.open(os.path.join(object_dir, names[similar[i][0]][len(image_dir)+1:].split('.')[0] + '.png')))
    plt.axis('off')
    plt.title("Second")

    fig.add_subplot(2, 2, 3)

    plt.imshow(Image.open(os.path.join(object_dir, names[similar[i][1]][len(image_dir)+1:].split('.')[0] + '.png')))
    plt.axis('off')
    plt.title("Second")

    fig.add_subplot(2, 2, 4)

    plt.imshow(Image.open(os.path.join(object_dir, names[similar[i][2]][len(image_dir)+1:].split('.')[0] + '.png')))
    plt.axis('off')
    plt.title("Second")

    # fig.add_subplot(2, 2, 5)
    # plt.imshow(Image.open(names[similar[i][3]]))
    # plt.axis('off')
    # plt.title("Second")

    # fig.add_subplot(2, 2, 6)
    # plt.imshow(Image.open(names[similar[i][4]]))
    # plt.axis('off')
    # plt.title("Second")
    
    plt.show()