import os
import shutil

root_dir = 'D:/Research/Counterfactual/Scripts/'

mainset = os.path.join(root_dir, 'original')
subset = os.path.join(root_dir, 'IN9sub')

for dirpath, dirnames, filenames in os.walk(mainset):
    structure1 = os.path.join(subset, dirpath[len(mainset) +1:])
    if not os.path.isdir(structure1):
        os.mkdir(structure1)

for dirpath, dirnames, filenames in os.walk(mainset):
        print(dirpath)
        if os.listdir(dirpath)[0].endswith('.JPEG'):
            i=0
            for i in range(400):
                # print(os.listdir(dirpath)[i])
                shutil.copyfile(os.path.join(dirpath, os.listdir(dirpath)[i]), os.path.join(subset, dirpath[len(mainset)+1:], os.listdir(dirpath)[i]))
                