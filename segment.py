from dis_loader import *
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from datetime import datetime

import os
import glob

import shutil
 
import os
 
from configs import *

if os.path.exists(object_dir):
    shutil.rmtree(object_dir)

if os.path.exists(bg_dir):
    shutil.rmtree(bg_dir)


if os.path.exists(mask_dir):
    shutil.rmtree(mask_dir)

os.mkdir(object_dir)
os.mkdir(bg_dir)
os.mkdir(mask_dir)

# then = datetime.now()
# now = datetime.now()
# print('Copied directory structure in ', now-then)

for dirpath, dirnames, filenames in os.walk(image_dir):
    structure1 = os.path.join(bg_dir, dirpath[len(image_dir) +1:])
    structure2 = os.path.join(object_dir, dirpath[len(image_dir) +1:])
    structure3 = os.path.join(mask_dir, dirpath[len(image_dir) +1:])

    if not os.path.isdir(structure1):
        os.mkdir(structure1)
    if not os.path.isdir(structure2):
        os.mkdir(structure2)
    if not os.path.isdir(structure3):
        os.mkdir(structure3)

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

i=0

for root, dirs, files in os.walk(image_dir):
	for img_name in files:
            image_path = os.path.join(root, img_name)
            image_tensor, orig_size = load_image(hypar=hypar, im_path=image_path)
            # image_tensor = Image.open(image_path)
            # image_tensor = np.array(image_tensor)
            print(image_tensor.shape)
            mask = predict(net,image_tensor,orig_size, hypar, device)
            # exit()

            image_mask = Image.fromarray(mask)
            image = Image.fromarray(image_path)
            blank = image.point(lambda _: 0)
            object = Image.composite(image, blank, image_mask)
            object = object.resize((image_dim,image_dim), Image.Resampling.LANCZOS)

            object.save(os.path.join(object_dir, root[len(image_dir) +1:], img_name.split('.')[0]+'.png'))
            
            image = image.resize((image_dim,image_dim), Image.Resampling.LANCZOS)
            image_mask = image_mask.resize((image_dim,image_dim), Image.Resampling.LANCZOS)

            image.save(os.path.join(bg_dir, root[len(image_dir) +1:], img_name.split('.')[0]+'.png'))
            image_mask.save(os.path.join(bg_dir, root[len(image_dir) +1:], img_name.split('.')[0]+'_mask001.png'))
            image_mask.save(os.path.join(mask_dir, root[len(image_dir) +1:], img_name.split('.')[0]+'.png'))

            i = i+1
            print(i)