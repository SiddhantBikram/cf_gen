import numpy as np
import os
from PIL import Image
from matplotlib import cm

arr = np.load('D:/Research/Counterfactual/Scripts/MNIST.npy', encoding='latin1', allow_pickle=True).item()

# test_dir = 'D:/Research/Counterfactual/Scripts/colored_mnist/test'
# train_dir = 'D:/Research/Counterfactual/Scripts/colored_mnist/train'

# for i in range(len(arr['train_label'])):    
#     try:
#         os.mkdir(os.path.join(train_dir, str(arr['train_label'][i])))
#     except:
#         pass

# for i in range(len(arr['train_label'])):    
#     im = Image.fromarray(arr['train_image'][i])
#     im.save(os.path.join(train_dir, str(arr['train_label'][i])) +'/' + str(i)+'.png')



# for i in range(len(arr['test_label'])):    
#     try:
#         os.mkdir(os.path.join(test_dir, str(arr['test_label'][i])))
#     except:
#         pass

# for i in range(len(arr['test_label'])):    
#     im = Image.fromarray(arr['test_image'][i])
#     im.save(os.path.join(test_dir, str(arr['test_label'][i])) +'/' + str(i)+'.png')

# # print(arr['train_image'].shape)