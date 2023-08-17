import os

root_dir = 'D:/Research/Counterfactual/Scripts/'
dataset_name = 'IN9sub'
image_dir = os.path.join(root_dir, dataset_name)
object_dir = os.path.join(root_dir, 'objects')
bg_dir = os.path.join(root_dir, 'bg')
mask_dir = os.path.join(root_dir, 'masks')
weight_dir =  os.path.join(root_dir, 'weights')
inpaint_dir = os.path.join(root_dir, 'inpaint')
bg_avg_dir = os.path.join(root_dir, 'bg_average')

image_dim = 256
