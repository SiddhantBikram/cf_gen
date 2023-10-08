import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from configs import *

im_1 = 'D:/Research/Counterfactual/Scripts/colored_mnist/train/2/'
# im_2 = 'D:/Research/Counterfactual/Scripts/colored_mnist/train/8'
img_collection = []

# for path in [im_1,im_2]:

for img in os.listdir(im_1):
    image = Image.open(os.path.join(im_1,img))
    arr = np.array(image)
    img_collection.append(arr)

X = np.ndarray((len(img_collection), np.product(img_collection[0].shape)))
for i, img in enumerate(img_collection):
    X[i] = img.reshape((-1)) # reshape((-1)) will 'straighten up' an n-d array into a 1-d array

pca = PCA(n_components=1000)
img_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, verbose=2)
img_tsne = tsne.fit_transform(img_pca)

# visualization example from https://github.com/ml4a/ml4a-guides/blob/master/notebooks/image-tsne.ipynb

width = 4000
height = 3000
max_dim = 100

# normalize the embedding so that it lies entirely in the range (0,1).
tx, ty = img_tsne[:,0], img_tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

full_image = Image.new('RGBA', (width, height))
for img, x, y in zip(img_collection, tx, ty):
    tile = Image.fromarray(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

plt.figure(figsize = (16,12))
full_image.save(os.path.join(root_dir, 'two.png'))

plt.imshow(full_image)


