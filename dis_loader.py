from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
import json
from tqdm import tqdm
from skimage import io
import os
from glob import glob
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from io import BytesIO
import requests

#### --------------------- DIS dataloader cache ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []
    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])

        # img_name_dict[im_dirs[i][0]] = tmp_im_list
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]

            # lbl_name_dict[im_dirs[i][0]] = tmp_gt_list
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


        if flag=="train": ## combine multiple training sets into one dataset
            if len(name_im_gt_list)==0:
                name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                        "im_path":tmp_im_list,
                                        "gt_path":tmp_gt_list,
                                        "im_ext":datasets[i]["im_ext"],
                                        "gt_ext":datasets[i]["gt_ext"],
                                        "cache_dir":datasets[i]["cache_dir"]})
            else:
                name_im_gt_list[0]["dataset_name"] = name_im_gt_list[0]["dataset_name"] + "_" + datasets[i]["name"]
                name_im_gt_list[0]["im_path"] = name_im_gt_list[0]["im_path"] + tmp_im_list
                name_im_gt_list[0]["gt_path"] = name_im_gt_list[0]["gt_path"] + tmp_gt_list
                if datasets[i]["im_ext"]!=".jpg" or datasets[i]["gt_ext"]!=".png":
                    print("Error: Please make sure all you images and ground truth masks are in jpg and png format respectively !!!")
                    exit()
                name_im_gt_list[0]["im_ext"] = ".jpg"
                name_im_gt_list[0]["gt_ext"] = ".png"
                name_im_gt_list[0]["cache_dir"] = os.sep.join(datasets[i]["cache_dir"].split(os.sep)[0:-1])+os.sep+name_im_gt_list[0]["dataset_name"]
        else: ## keep different validation or inference datasets as separate ones
            name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                    "im_path":tmp_im_list,
                                    "gt_path":tmp_gt_list,
                                    "im_ext":datasets[i]["im_ext"],
                                    "gt_ext":datasets[i]["gt_ext"],
                                    "cache_dir":datasets[i]["cache_dir"]})

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, cache_size=[], cache_boost=True, my_transforms=[], batch_size=1, shuffle=False):
    ## model="train": return one dataloader for training
    ## model="valid": return a list of dataloaders for validation or testing

    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8

    for i in range(0,len(name_im_gt_list)):
        gos_dataset = GOSDatasetCache([name_im_gt_list[i]],
                                      cache_size = cache_size,
                                      cache_path = name_im_gt_list[i]["cache_dir"],
                                      cache_boost = cache_boost,
                                      transform = transforms.Compose(my_transforms))
        gos_dataloaders.append(DataLoader(gos_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers_))
        gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

def im_reader(im_path):
    return io.imread(im_path)

def im_preprocess(im,size):
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    im_tensor = torch.tensor(im.copy(), dtype=torch.float32)
    im_tensor = torch.transpose(torch.transpose(im_tensor,1,2),0,1)
    if(len(size)<2):
        return im_tensor, im.shape[0:2]
    else:
        im_tensor = torch.unsqueeze(im_tensor,0)
        im_tensor = F.upsample(im_tensor, size, mode="bilinear")
        im_tensor = torch.squeeze(im_tensor,0)

    return im_tensor.type(torch.uint8), im.shape[0:2]

def gt_preprocess(gt,size):
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    gt_tensor = torch.unsqueeze(torch.tensor(gt, dtype=torch.uint8),0)

    if(len(size)<2):
        return gt_tensor.type(torch.uint8), gt.shape[0:2]
    else:
        gt_tensor = torch.unsqueeze(torch.tensor(gt_tensor, dtype=torch.float32),0)
        gt_tensor = F.upsample(gt_tensor, size, mode="bilinear")
        gt_tensor = torch.squeeze(gt_tensor,0)

    return gt_tensor.type(torch.uint8), gt.shape[0:2]
    # return gt_tensor, gt.shape[0:2]

class GOSRandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class GOSResize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # import time
        # start = time.time()

        image = torch.squeeze(F.upsample(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        label = torch.squeeze(F.upsample(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)

        # print("time for resize: ", time.time()-start)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class GOSRandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}


class GOSNormalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}


class GOSDatasetCache(Dataset):

    def __init__(self, name_im_gt_list, cache_size=[], cache_path='./cache', cache_file_name='dataset.json', cache_boost=False, transform=None):


        self.cache_size = cache_size
        self.cache_path = cache_path
        self.cache_file_name = cache_file_name
        self.cache_boost_name = ""

        self.cache_boost = cache_boost
        # self.ims_npy = None
        # self.gts_npy = None

        ## cache all the images and ground truth into a single pytorch tensor
        self.ims_pt = None
        self.gts_pt = None

        ## we will cache the npy as well regardless of the cache_boost
        # if(self.cache_boost):
        self.cache_boost_name = cache_file_name.split('.json')[0]

        self.transform = transform

        self.dataset = {}

        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_shp"] = []
        self.dataset["gt_shp"] = []
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list


        self.dataset["ims_pt_dir"] = ""
        self.dataset["gts_pt_dir"] = ""

        self.dataset = self.manage_cache(dataset_names)

    def manage_cache(self,dataset_names):
        if not os.path.exists(self.cache_path): # create the folder for cache
            os.makedirs(self.cache_path)
        cache_folder = os.path.join(self.cache_path, "_".join(dataset_names)+"_"+"x".join([str(x) for x in self.cache_size]))
        if not os.path.exists(cache_folder): # check if the cache files are there, if not then cache
            return self.cache(cache_folder)
        return self.load_cache(cache_folder)

    def cache(self,cache_folder):
        os.mkdir(cache_folder)
        cached_dataset = deepcopy(self.dataset)

        # ims_list = []
        # gts_list = []
        ims_pt_list = []
        gts_pt_list = []
        for i, im_path in tqdm(enumerate(self.dataset["im_path"]), total=len(self.dataset["im_path"])):

            im_id = cached_dataset["im_name"][i]
            print("im_path: ", im_path)
            im = im_reader(im_path)
            im, im_shp = im_preprocess(im,self.cache_size)
            im_cache_file = os.path.join(cache_folder,self.dataset["data_name"][i]+"_"+im_id + "_im.pt")
            torch.save(im,im_cache_file)

            cached_dataset["im_path"][i] = im_cache_file
            if(self.cache_boost):
                ims_pt_list.append(torch.unsqueeze(im,0))
            # ims_list.append(im.cpu().data.numpy().astype(np.uint8))

            gt = np.zeros(im.shape[0:2])
            if len(self.dataset["gt_path"])!=0:
                gt = im_reader(self.dataset["gt_path"][i])
            gt, gt_shp = gt_preprocess(gt,self.cache_size)
            gt_cache_file = os.path.join(cache_folder,self.dataset["data_name"][i]+"_"+im_id + "_gt.pt")
            torch.save(gt,gt_cache_file)
            if len(self.dataset["gt_path"])>0:
                cached_dataset["gt_path"][i] = gt_cache_file
            else:
                cached_dataset["gt_path"].append(gt_cache_file)
            if(self.cache_boost):
                gts_pt_list.append(torch.unsqueeze(gt,0))
            # gts_list.append(gt.cpu().data.numpy().astype(np.uint8))

            # im_shp_cache_file = os.path.join(cache_folder,im_id + "_im_shp.pt")
            # torch.save(gt_shp, shp_cache_file)
            cached_dataset["im_shp"].append(im_shp)
            # self.dataset["im_shp"].append(im_shp)

            # shp_cache_file = os.path.join(cache_folder,im_id + "_gt_shp.pt")
            # torch.save(gt_shp, shp_cache_file)
            cached_dataset["gt_shp"].append(gt_shp)
            # self.dataset["gt_shp"].append(gt_shp)

        if(self.cache_boost):
            cached_dataset["ims_pt_dir"] = os.path.join(cache_folder, self.cache_boost_name+'_ims.pt')
            cached_dataset["gts_pt_dir"] = os.path.join(cache_folder, self.cache_boost_name+'_gts.pt')
            self.ims_pt = torch.cat(ims_pt_list,dim=0)
            self.gts_pt = torch.cat(gts_pt_list,dim=0)
            torch.save(torch.cat(ims_pt_list,dim=0),cached_dataset["ims_pt_dir"])
            torch.save(torch.cat(gts_pt_list,dim=0),cached_dataset["gts_pt_dir"])

        try:
            json_file = open(os.path.join(cache_folder, self.cache_file_name),"w")
            json.dump(cached_dataset, json_file)
            json_file.close()
        except Exception:
            raise FileNotFoundError("Cannot create JSON")
        return cached_dataset

    def load_cache(self, cache_folder):
        json_file = open(os.path.join(cache_folder,self.cache_file_name),"r")
        dataset = json.load(json_file)
        json_file.close()
        ## if cache_boost is true, we will load the image npy and ground truth npy into the RAM
        ## otherwise the pytorch tensor will be loaded
        if(self.cache_boost):
            # self.ims_npy = np.load(dataset["ims_npy_dir"])
            # self.gts_npy = np.load(dataset["gts_npy_dir"])
            self.ims_pt = torch.load(dataset["ims_pt_dir"], map_location='cpu')
            self.gts_pt = torch.load(dataset["gts_pt_dir"], map_location='cpu')
        return dataset

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):

        im = None
        gt = None
        if(self.cache_boost and self.ims_pt is not None):

            # start = time.time()
            im = self.ims_pt[idx]#.type(torch.float32)
            gt = self.gts_pt[idx]#.type(torch.float32)
            # print(idx, 'time for pt loading: ', time.time()-start)

        else:
            # import time
            # start = time.time()
            # print("tensor***")
            im_pt_path = os.path.join(self.cache_path,os.sep.join(self.dataset["im_path"][idx].split(os.sep)[-2:]))
            im = torch.load(im_pt_path)#(self.dataset["im_path"][idx])
            gt_pt_path = os.path.join(self.cache_path,os.sep.join(self.dataset["gt_path"][idx].split(os.sep)[-2:]))
            gt = torch.load(gt_pt_path)#(self.dataset["gt_path"][idx])
            # print(idx,'time for tensor loading: ', time.time()-start)


        im_shp = self.dataset["im_shp"][idx]
        # print("time for loading im and gt: ", time.time()-start)

        # start_time = time.time()
        im = torch.divide(im,255.0)
        gt = torch.divide(gt,255.0)
        # print(idx, 'time for normalize torch divide: ', time.time()-start_time)

        sample = {
        "imidx": torch.from_numpy(np.array(idx)),
        "image": im,
        "label": gt,
        "shape": torch.from_numpy(np.array(im_shp)),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


bce_loss = nn.BCELoss(size_average=True)
def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0,len(preds)):
        # print("i: ", i, preds[i].shape)
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target)
        if(i==0):
            loss0 = loss
    return loss0, loss

fea_loss = nn.MSELoss(size_average=True)
kl_loss = nn.KLDivLoss(size_average=True)
l1_loss = nn.L1Loss(size_average=True)
smooth_l1_loss = nn.SmoothL1Loss(size_average=True)
def muti_loss_fusion_kl(preds, target, dfs, fs, mode='MSE'):
    loss0 = 0.0
    loss = 0.0

    for i in range(0,len(preds)):
        # print("i: ", i, preds[i].shape)
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target)
        if(i==0):
            loss0 = loss

    for i in range(0,len(dfs)):
        if(mode=='MSE'):
            loss = loss + fea_loss(dfs[i],fs[i]) ### add the mse loss of features as additional constraints
            # print("fea_loss: ", fea_loss(dfs[i],fs[i]).item())
        elif(mode=='KL'):
            loss = loss + kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1))
            # print("kl_loss: ", kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1)).item())
        elif(mode=='MAE'):
            loss = loss + l1_loss(dfs[i],fs[i])
            # print("ls_loss: ", l1_loss(dfs[i],fs[i]))
        elif(mode=='SmoothL1'):
            loss = loss + smooth_l1_loss(dfs[i],fs[i])
            # print("SmoothL1: ", smooth_l1_loss(dfs[i],fs[i]).item())

    return loss0, loss

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1,stride=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate,stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU7,self).__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1) ## 1 -> 1/2

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        b, c, h, w = x.shape

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


class myrebnconv(nn.Module):
    def __init__(self, in_ch=3,
                       out_ch=1,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       dilation=1,
                       groups=1):
        super(myrebnconv,self).__init__()

        self.conv = nn.Conv2d(in_ch,
                              out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.rl = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.rl(self.bn(self.conv(x)))


class ISNetGTEncoder(nn.Module):

    def __init__(self,in_ch=1,out_ch=1):
        super(ISNetGTEncoder,self).__init__()

        self.conv_in = myrebnconv(in_ch,16,3,stride=2,padding=1) # nn.Conv2d(in_ch,64,3,stride=2,padding=1)

        self.stage1 = RSU7(16,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,32,128)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(128,32,256)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(256,64,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,64,512)


        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

    def compute_loss(self, preds, targets):

        return muti_loss_fusion(preds,targets)

    def forward(self,x):

        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        #stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)


        #side output
        d1 = self.side1(hx1)
        d1 = _upsample_like(d1,x)

        d2 = self.side2(hx2)
        d2 = _upsample_like(d2,x)

        d3 = self.side3(hx3)
        d3 = _upsample_like(d3,x)

        d4 = self.side4(hx4)
        d4 = _upsample_like(d4,x)

        d5 = self.side5(hx5)
        d5 = _upsample_like(d5,x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,x)

        # d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)], [hx1,hx2,hx3,hx4,hx5,hx6]

class ISNetDIS(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(ISNetDIS,self).__init__()

        self.conv_in = nn.Conv2d(in_ch,64,3,stride=2,padding=1)
        self.pool_in = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage1 = RSU7(64,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        # self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def compute_loss_kl(self, preds, targets, dfs, fs, mode='MSE'):

        # return muti_loss_fusion(preds,targets)
        return muti_loss_fusion_kl(preds, targets, dfs, fs, mode=mode)

    def compute_loss(self, preds, targets):

        # return muti_loss_fusion(preds,targets)
        return muti_loss_fusion(preds, targets)

    def forward(self,x):

        hx = x

        hxin = self.conv_in(hx)
        #hx = self.pool_in(hxin)

        #stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1,x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,x)

        # d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)],[hx1d,hx2d,hx3d,hx4d,hx5d,hx6]


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image


transform =  transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])

def load_image(im_path, hypar, image_dim):
    if im_path.startswith("http"):
        im_path = BytesIO(requests.get(im_path).content)

    im = im_reader(im_path)
    # im = resize(im, (image_dim, image_dim), anti_aliasing=True)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im,255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0) # make a batch of image, shape

def build_model(hypar,device):
    net = hypar["model"]#GOSNETINC(3,1)

    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if(hypar["restore_model"]!=""):
        net.load_state_dict(torch.load(hypar["model_path"]+"/"+hypar["restore_model"],map_location=device))
        net.to(device)
    net.eval()
    return net


def predict(net,  inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if(hypar["model_digit"]=="full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)


    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device) # wrap inputs in Variable

    ds_val = net(inputs_val_v)[0] # list of 6 results

    pred_val = ds_val[0][0,:,:,:] # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[0][0],shapes_val[0][1]),mode='bilinear'))


    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi) # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8) # it is the mask we need



