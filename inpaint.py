import sys
import os
from configs import *
sys.path.append(os.path.join(root_dir,'cf_gen','lama'))

import logging
import os
import sys
import traceback


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import glob
import shutil

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device

from configs import *

inpaint_dir = os.path.join(root_dir, 'inpaint')
bg_dir = os.path.join(root_dir, 'bg')

if os.path.exists(inpaint_dir):
    shutil.rmtree(inpaint_dir)

os.mkdir(inpaint_dir)

for dirpath, dirnames, filenames in os.walk(bg_dir):
    structure1 = os.path.join(inpaint_dir, dirpath[len(bg_dir) +1:])
    if not os.path.isdir(structure1):
        os.mkdir(structure1)

@hydra.main(config_path=os.path.join(root_dir,'cf_gen/lama/configs/prediction'), config_name='inpaint_test.yaml')
def main(predict_config: OmegaConf):

    device = 'cuda'

    train_config_path = os.path.join(root_dir, 'cf_gen/lama/configs/prediction/inpaint_train.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    out_ext = predict_config.get('out_ext', '.png')

    checkpoint_path = os.path.join(weight_dir, 'inpaint_weights.ckpt')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    i=0
    for dirpath, dirnames, filenames in os.walk(bg_dir):
        if os.listdir(dirpath)[0].endswith('.png'):
            dataset = make_default_val_dataset(dirpath, **predict_config.dataset)

            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                #Inpaint directory, subdirectory, and image name respectively
                cur_out_fname = os.path.join(inpaint_dir, dirpath[len(bg_dir) +1:], mask_fname.split('\\')[-1].split('.')[0][:-8]+'.png')
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
                batch = default_collate([dataset[img_i]])
                print(dataset[img_i].keys())

                # if predict_config.get('refine', False):
                #     assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                #     cur_res = refine_predict(batch, model, **predict_config.refiner)
                #     cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
                # else:

                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)
                print(i)
                i = i+1

main()
