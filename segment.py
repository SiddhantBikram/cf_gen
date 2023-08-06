from dis_loader import *
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

import os
import glob

root_dir = 'D:/Research/Counterfactual/Scripts/'

object_dir = os.path.join(root_dir, 'objects')
bg_dir = os.path.join(root_dir, 'bg')

files = glob.glob(object_dir + '/*') + glob.glob(bg_dir +'/*')
for f in files:
    os.remove(f)

def seg_model(size, seed):
    hypar = {} # paramters for inferencing
    hypar["model_path"] = os.path.join(root_dir, 'weights') ## load trained weights from this path
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

image_dir = os.path.join(root_dir, 'imagenet-mini')

i=0
    
for root, dirs, files in os.walk(image_dir):
	for img_name in files:
            # print(img_name)
            image_path = os.path.join(root, img_name)
            image_tensor, orig_size = load_image(image_path, hypar)
            mask = predict(net,image_tensor,orig_size, hypar, device)
            image_mask = Image.fromarray(mask)
            image = Image.open(image_path)
            blank = image.point(lambda _: 0)
            object = Image.composite(image, blank, image_mask)

            # mask_invert = ImageOps.invert(image_mask)

            object.save(os.path.join(object_dir, img_name))
            
            # print(os.path.join(bg_dir, img_name.split('.')[0]+'.png'))
            image.save(os.path.join(bg_dir, img_name.split('.')[0]+'.png'))
            image_mask.save(os.path.join(bg_dir, img_name.split('.')[0]+'_mask001.png'))
            i = i+1