import os

dataset_name = 'IN9sub'
n_classes = 9

root_dir = '/content/drive/MyDrive/Scripts'
image_dir = '/content/drive/MyDrive/Scripts/cf_gen/colored_mnist/mnist-lt'
object_dir = os.path.join(root_dir, 'objects')
bg_dir = os.path.join(root_dir, 'bg')
mask_dir = os.path.join(root_dir, 'masks')
weight_dir =  os.path.join(root_dir, 'weights')
inpaint_dir = os.path.join(root_dir, 'inpaint')
bg_avg_dir = os.path.join(root_dir, 'bg_average')
train_dir = os.path.join(image_dir, 'train')
val_dir = os.path.join(image_dir, 'val')

device = 'cuda'
image_dim = 256
n_clusters = n_classes
seed = 510