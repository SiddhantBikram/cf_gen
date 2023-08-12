from pfd_loader import *
import os 
from PIL import Image
import sys

root_dir = 'D:/Research/Counterfactual/Scripts/'
sys.path.append(os.path.join(root_dir,'cf_gen','pfd'))


#Input Image = Class
#Control Image = Second most confident class among all samples

pfd_inference = prompt_free_diffusion(
    fp16=True, tag_ctx = 'SeeCoder', tag_diffuser = 'Deliberate-v2.0', tag_ctl = 'canny_v11p')

cache_examples = True
image_dim = 256

image_input = Image.open("D:/Research/Counterfactual/Scripts/IN9sub/train/00_dog/n02085620_7.JPEG")
control_image = Image.open("D:/Research/Counterfactual/Scripts/objects/train/00_dog/n02085620_1321.png")

out_image = pfd_inference.action_inference(im = image_input,
                                           imctl = control_image,
                                            do_preprocess = False, 
                                            h = image_dim, 
                                            w = image_dim,
                                            ugscale = 0,
                                            seed = 42, 
                                            tag_ctx = 'SeeCoder',
                                            tag_diffuser = 'Deliberate-v2.0',
                                            tag_ctl = 'canny_v11p')