from datasets import load_dataset
from PIL import Image
import os

dataset = load_dataset("mnist")

print(dataset)

# test_dir = 'D:/Research/Counterfactual/Scripts/CIFAR-10 LT/train'
# train_dir = 'D:/Research/Counterfactual/Scripts/CIFAR-10 LT/test'
dir = 'D:/Research/Counterfactual/Scripts/MNIST'

# for i in ['train', 'test']:
#     for j in range(60000):
#         try:
#             os.mkdir(os.path.join(dir, i, str(dataset[i][j]['label'])))
#         except:
#             pass


for i in ['train', 'test']:
    for j in range(60000):
        try:
            dataset[i][j]['image'].save(os.path.join(dir, i, str(dataset[i][j]['label']), str(j)+'.png'))
        except:
            pass

