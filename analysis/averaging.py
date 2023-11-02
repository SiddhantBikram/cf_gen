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

    
    return avg_image

def stackImagesKeypointMatching(dirpath):

    orb = cv2.ORB_create()

    file_list = os.listdir(dirpath)

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)
        # compute the descriptors with ORB
    i=0
    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for file in file_list:
        image = cv2.imread(os.path.join(dirpath,file),1)
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
             # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            try:
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                w, h, _ = imageF.shape
                imageF = cv2.warpPerspective(imageF, M, (h, w))
                stacked_image += imageF
            except:
                pass  
            i = i+1

            if i ==5:
                break

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image

for dirpath, dirnames, filenames in os.walk(inpaint_dir):
    if os.listdir(dirpath)[0].endswith('.png'):
        out = stackImagesKeypointMatching(dirpath)
        plt.imshow(out)
        plt.show()
        save_path = os.path.join(bg_avg_dir, dirpath[len(inpaint_dir) +1:], dirpath.split('\\')[-1]+'.png')
        cv2.imwrite(save_path, out)
