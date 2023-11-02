import os
import matplotlib.pyplot as plt

dir = 'D:/Research/Counterfactual/Scripts/MNIST-LT/train'

x = os.listdir(dir)
y = [len(os.listdir(os.path.join(dir,i))) for i in x]
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.bar(x, y)
plt.show()