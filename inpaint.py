import sys
import os

root_dir = 'D:/Research/Counterfactual/Scripts/'

sys.path.append(os.path.join(root_dir,'cf_gen','lama'))
# sys.path.append(os.path.join(root_dir,'cf_gen','lama', 'models'))

# os.chdir(os.path.join(root_dir,'cf_gen','lama'))

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
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

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

root_dir = 'D:/Research/Counterfactual/Scripts/'

inpaint_dir = os.path.join(root_dir, 'inpaint')

files = glob.glob(inpaint_dir + '/*')
for f in files:
    os.remove(f)

@hydra.main(config_path=os.path.join(root_dir,'cf_gen/lama/configs/prediction'), config_name='inpaint_test.yaml')
def main(predict_config: OmegaConf):
    try:
        # register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(root_dir, 'cf_gen/lama/configs/prediction/inpaint_train.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(root_dir, 'weights', 'inpaint_weights.ckpt')
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        # if not predict_config.indir.endswith('/'):
        #     predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = 'D:/Research/Counterfactual/Scripts/inpaint'+os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-8]+ out_ext
            
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])

            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:

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

    except KeyboardInterrupt:
        print('1')



main()

# model: torch.nn.Module = make_training_model(train_config)
# state = torch.load(path, map_location=map_location)
# model.load_state_dict(state['state_dict'], strict=strict)
# model.on_load_checkpoint(state)