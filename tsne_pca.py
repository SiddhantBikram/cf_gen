import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from configs import *
from torchvision import models
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from dis_loader import *

dir = train_dir
img_collection = []

def segment(path, seg_model, seg_hypar):

    image_tensor, orig_size = load_image(hypar=seg_hypar, im_path=path)
    mask = predict(seg_model,image_tensor,orig_size, seg_hypar, device)

    mask = Image.fromarray(mask)
    mask = mask.resize((image_dim,image_dim), Image.Resampling.LANCZOS)

    image = Image.open(path)
    image = image.resize((image_dim,image_dim), Image.Resampling.LANCZOS)
    blank = image.point(lambda _: 0)
    obj = Image.composite(image, blank, mask)
    obj = obj.resize((image_dim,image_dim), Image.Resampling.LANCZOS)

    return obj, mask

def main():

    seg_model, seg_hypar = init_seg(1024, seed)

    resnet = models.resnet50(pretrained=True).to(device)
    resnet.state_dict(torch.load(os.path.join(root_dir, 'weights', dataset_name, 'ssl_encoder.pt')))

    trans = transforms.Compose([transforms.ToTensor()])
    X = []

    resnet.eval()

    for folder in os.listdir(dir):
        for im in os.listdir(os.path.join(dir,folder)):
            with torch.no_grad():
                obj, mask = segment(os.path.join(dir,folder,im), seg_model, seg_hypar)
                # image = Image.open(os.path.join(dir,folder,im))

                y_nonzero, x_nonzero, _ = np.nonzero(obj)
                obj = obj.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))
                obj= obj.resize((image_dim,image_dim))

                arr = np.array(obj)
                img_collection.append(arr)

                encoded = resnet(trans(obj).unsqueeze(0).to(device))
                X.append(encoded.cpu().numpy().squeeze(0))


    X = np.array(X)
    print(X.shape)

    pca = PCA(n_components=900)
    img_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, verbose=2)
    img_tsne = tsne.fit_transform(img_pca)


    width = 4000
    height = 3000
    max_dim = 100

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
    full_image.save(os.path.join(root_dir, dataset_name+'.png'))

    plt.imshow(full_image)

if __name__ == "__main__":
    main()

    