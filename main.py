import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import os
import wandb
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
import warnings

import config
from dataloader import *
from backdoor import *
from encoder import *
from uncertainty import *
from contrastive import *
from generator import *
import helper 

num_classes = helper.get_class_num(config.dataset) 

np.set_printoptions(3, suppress=True)
device = torch.device('cuda')

seed = config.seed
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

val_metric = 'val_acc'
num_classes = helper.get_class_num(config.dataset) 


#BYOL


print('SSL Completed.')


train_dataset = Dataset(dataset=config.dataset, mode='train', num_classifier=config.num_classifier, set_size=config.set_size)
train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )
save_train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
valid_dataset = Dataset(dataset=config.dataset, mode='valid')
valid_loader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
test_dataset = Dataset(dataset=config.dataset, mode='test')
test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

f_dim = 512

class Encoder(torch.nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = timm.create_model('resnet101', pretrained=True).to(device)
        self.lin = nn.Linear(x, f_dim)

        def forward(self, x):
            x = self.encoder(x)
            x = self.lin(x)

            return x

classifiers = []

for _ in range(config.num_classifier):
    classifiers.append(helper.get_classifier(config.classifier, config.f_dim, config.output_dim, config.linear_bias).to(config.device))

params = []
for i in range(config.num_classifier):
    params += list(classifiers[i].parameters())

target_classifier = helper.get_classifier(config.target_classifier, f_dim, output_dim, config.linear_bias).to(device)

if config.optimizer =='SGD':
    print("SGD")
    optimizer=torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    optimizer_target=torch.optim.SGD(target_classifier.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
else:
    optimizer=torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    optimizer_target=torch.optim.Adam(target_classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)

kl_loss = torch.nn.KLDivLoss(reduce=False)
best_metric = {}
best_metric[val_metric] = 0

for epoch in range(config.epochs):

    losses = AverageMeter()
    target_losses = AverageMeter()
    kd_losses = AverageMeter()
    ens_losses = AverageMeter()

    for i in range(config.num_classifier):
        classifiers[i].train()

    mask_sum = torch.Tensor([0]*config.num_classifier)
    weights = []
    num_samples = 0
    tot_correct = torch.Tensor([0]*(config.num_classifier+1))
    correct_split = torch.zeros(config.num_classifier+1, config.num_classes)
    count_split = torch.zeros(config.num_classifier+1, config.num_classes)

    for data, labels, sensitive, mask in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        data = data.to(device)
        labels = labels.to(device)
        loss = 0
        num_samples += data.shape[0]
        outputs =[]
        loss_cand = []

        for i in range(config.num_classifier):
            outputs.append(classifiers[i](data)*config.temperature)   
            tot_correct[i] += num_correct(outputs[i], labels).item()
            s_correct, s_count =  helper.num_correct_cls(outputs[i],labels, sensitive, config.output_dim, 1)
            correct_split[i] += s_correct
            count_split[i] += s_count 
            count = mask[i].sum()
            if count>0:
                loss += (ce(outputs[i], labels, False)*mask[i].to(device)).sum()/mask[i].sum()