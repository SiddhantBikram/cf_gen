import os


i = 1

dir = 'D:/Research/Counterfactual/Scripts/ImageNet-Subset/train'

for folder in os.listdir(dir):
    for im in os.listdir(os.path.join(dir,folder))[i*100:]:
        os.remove(os.path.join(dir, folder, im))
    
    i+=1
        