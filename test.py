import glob
import os

# root_dir = 'D:/Research/Counterfactual/Scripts/'

# print(glob.glob('D:/Research/Counterfactual/Scripts/imagenet-mini/train'))
filelist = []

for root, dirs, files in os.walk('D:/Research/Counterfactual/Scripts/imagenet-mini/train'):
	for file in files:
        #append the file name to the list
		filelist.append(os.path.join(root,file))
		

print(filelist)