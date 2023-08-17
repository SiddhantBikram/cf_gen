from configs import *
import numpy as np
from PIL import Image
import shutil
import cv2
import matplotlib.pyplot as plt

if os.path.exists(bg_avg_dir):
    shutil.rmtree(bg_avg_dir)

os.mkdir(bg_avg_dir)

for dirpath, dirnames, filenames in os.walk(inpaint_dir):
    structure1 = os.path.join(bg_avg_dir, dirpath[len(inpaint_dir) +1:])

    if not os.path.isdir(structure1):
        os.mkdir(structure1)

def bg_avg(dirpath):
    arr=np.zeros((image_dim,image_dim,3),np.float64)

    image_list = os.listdir(dirpath)
    i = 0
    avg_image = cv2.imread(os.path.join(dirpath,image_list[0]))
    for i in range(len(image_list)):
        if i==0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            img = cv2.imread(os.path.join(dirpath,image_list[i]))
            avg_image = cv2.addWeighted(img, alpha, avg_image, beta, 0.0)

            if i ==5:
                break

    plt.imshow(avg_image)
    plt.show()
    return avg_image

for dirpath, dirnames, filenames in os.walk(inpaint_dir):
    if os.listdir(dirpath)[0].endswith('.png'):
        out = bg_avg(dirpath)
        save_path = os.path.join(bg_avg_dir, dirpath[len(inpaint_dir) +1:], dirpath.split('\\')[-1]+'.png')
        cv2.imwrite(save_path, out)
