from dis_loader import *
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

import torch
from torchvision import transforms
import models

import os
import glob
import yaml

import warnings
warnings.filterwarnings("ignore")

object_dir = 'C:/Users/siddh/Desktop/Counterfactual/Scripts/objects'
bg_dir = 'C:/Users/siddh/Desktop/Counterfactual/Scripts/bg'
image_dir = "D:/Research/imagenet-mini/train/n01440764"

files = glob.glob(object_dir+ '/*') + glob.glob(bg_dir+ '/*')

for f in files:
    os.remove(f)

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))

def seg_model(size, seed):
    hypar = {} # paramters for inferencing
    hypar["model_path"] ="C:/Users/siddh/Desktop/Counterfactual/Scripts/weights" ## load trained weights from this path
    hypar["restore_model"] = "seg_weight.pth" ## name of the to-be-loaded weights
    hypar["interm_sup"] = False ## indicate if activate intermediate feature supervision
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = seed
    hypar["cache_size"] = [size, size] ## cached input spatial resolution, can be configured into different size
    hypar["input_size"] = [size, size] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
    hypar["crop_size"] = [size, size] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation
    hypar["model"] = ISNetDIS()

    net = build_model(hypar, device)

    return net, hypar

net, hypar = seg_model(1024, 42)


i=0

with open('C:/Users/siddh/Desktop/Counterfactual/Scripts/cf_gen/yaml.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

inpaint_model = models.make(config['model']).cuda()
inpaint_model.encoder.load_state_dict(torch.load('C:/Users/siddh/Desktop/Counterfactual/Scripts/weights/encoder-epoch-last.pth', map_location='cuda:0'))

h=256
w=256

for img_name in os.listdir(image_dir):
    print(i)
    image_path = image_dir + '/' +img_name
    image_tensor, orig_size = load_image(image_path, hypar)
    mask = predict(net,image_tensor,orig_size, hypar, device)
    image_mask = Image.fromarray(mask)
    image = Image.open(image_path)
    blank = image.point(lambda _: 0)
    object = Image.composite(image, blank, image_mask)

    mask_invert = ImageOps.invert(image_mask)
    mask_invert.save(os.path.join(bg_dir, image_path.split('/')[-1]))
    mask_invert = transforms.ToTensor()(Image.open(os.path.join(bg_dir, image_path.split('/')[-1])).convert('RGB'))
    img = transforms.ToTensor()(Image.open(image_path).convert('RGB'))

    img = resize_fn(img, (h, w))
    # img = (img - 0.5) / 0.5
    mask_invert = resize_fn(mask_invert, (h, w))
    mask_invert = to_mask(mask_invert)
    # mask_invert[mask_invert > 0] = 1
    # mask_invert = 1 - mask_invert

    with torch.no_grad():
        bg = inpaint_model.encoder.mask_predict([img.unsqueeze(0).cuda(), mask_invert.unsqueeze(0).cuda()])
    bg = transforms.ToPILImage()(bg.squeeze(0))
    # bg = Image.composite(image, blank, mask_invert)

    object.save(os.path.join(object_dir, image_path.split('/')[-1]))
    bg.save(os.path.join(bg_dir, image_path.split('/')[-1]))
    i = i+1