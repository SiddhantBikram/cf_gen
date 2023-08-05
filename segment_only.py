from dis_loader import *
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

import os
import glob

object_dir = 'C:/Users/siddh/Desktop/Counterfactual/Scripts/objects'
bg_dir = 'C:/Users/siddh/Desktop/Counterfactual/Scripts/bg'

# files = glob.glob(object_dir) + glob.glob(bg_dir)
# for f in files:
#     os.remove(f)

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

image_dir = "C:/Users/siddh/Desktop/Counterfactual/Scripts/MNIST-M/testing/0"

i=0

for img_name in os.listdir(image_dir):
    print(i)
    image_path = image_dir + '/' +img_name
    image_tensor, orig_size = load_image(image_path, hypar)
    mask = predict(net,image_tensor,orig_size, hypar, device)
    image_mask = Image.fromarray(mask)
    image = Image.open(image_path)
    blank = image.point(lambda _: 0)
    im_invert = ImageOps.invert(image_mask)
    object = Image.composite(image, blank, image_mask)
    bg = Image.composite(image, blank, im_invert)
    object.save(object_dir+ image_path.split('/')[-1])
    bg.save(bg_dir+ image_path.split('/')[-1])
    i = i+1