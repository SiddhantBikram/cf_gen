import sys
import os
from configs import *
sys.path.append(os.path.join(root_dir,'cf_gen','lama'))
from omegaconf import OmegaConf
import os
import sys
import yaml
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
import torch
import numpy as np
import cv2

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.append(os.path.join(root_dir,'cf_gen','lama'))

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device

def init_inpaint():
    
    train_config_path = os.path.join(weight_dir, 'inpaint_config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(weight_dir, 'inpaint_weights.ckpt')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)

    return model

def move_to_device(obj, device):
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')

def inpaint(img, mask, inpaint_model):

    tuple = {}
    tuple['image'] = np.array(img)
    tuple['mask'] = np.expand_dims(np.array(mask), axis=0)
    tuple['unpad_to_size'] = [torch.Tensor([image_dim]), torch.Tensor([image_dim])]
    batch = default_collate([tuple])

    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = inpaint_model(batch)                    
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

    bg = np.clip(cur_res * 255, 0, 255).astype('uint8')
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    return bg