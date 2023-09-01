import torch
from byol_loader import BYOL, TwoCropsTransform, GaussianBlur
from lama_loader import load_checkpoint, move_to_device
from torchvision import models
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import warnings
import torchvision.datasets as datasets
import pickle
import tqdm
warnings.filterwarnings("ignore")

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from torch.utils.data._utils.collate import default_collate
import cv2
from omegaconf import OmegaConf
import yaml
from segment import build_model

from backdoor_loader import *
from configs import *
from dis_loader import *

epochs = 4
lr = 1e-3
gamma = 0.7
seed = 510

augmentation = [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]

train_dataset = datasets.ImageFolder(train_dir, TwoCropsTransform(transforms.Compose(augmentation)))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle = True, pin_memory=True, drop_last=True)
val_dataset = datasets.ImageFolder(val_dir)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle = True, pin_memory=True, drop_last=True)

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

def seg_model(size, seed):
    hypar = {} # paramters for inferencing
    hypar["model_path"] = os.path.join(root_dir, 'weights') ## load trained weights from this path
    hypar["restore_model"] = "seg_weight.pth" ## name of the to-be-loaded weights
    hypar["interm_sup"] = False ## indicate if activate intermediate feature supervision
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = seed
    hypar["cache_size"] = [size, size] ## cached input spatial resolution, can be configured into different size
    hypar["input_size"] = [size, size] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
    hypar["crop_size"] = [size, size] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation
    hypar["model"] = ISNetDIS()

    net = build_model(hypar, device)

    return net, hypar

def init_inpaint():
    
    train_config_path = os.path.join(root_dir, 'cf_gen/lama/configs/prediction/inpaint_train.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(weight_dir, 'inpaint_weights.ckpt')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)

    return model

class classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()

        # self.fc1 = nn.Linear(512, 128)
        # self.fc2 = nn.Linear(128, n_classes)
        self.fc = nn.Linear(image_dim*2, n_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self, concat):
      
    #   linear_out = self.dropout(F.relu(self.fc1(concat)))    

    #   final_out = self.fc2(linear_out)   

        final_out = self.fc(concat)

        return final_out

def train_ssl(model, load_weights = True):

    if load_weights:
        model.state_dict(torch.load(os.path.join(weight_dir, dataset_name, 'resnet.pt')))

        return model

    learner = BYOL(
        model,
        image_size = image_dim,
        hidden_layer = 'avgpool',
        return_embedding = False
    )

    opt = torch.optim.Adam(learner.parameters(), lr=1e-5)

    learner.train()

    for i in range(epochs):
        for j, (images, _) in enumerate(train_loader):

            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)

            loss = learner(image_one=images[0], image_two=images[1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() 
        print("Epoch: ",i," Loss: ", loss.item())
    torch.save(model.state_dict(),  os.path.join(weight_dir, dataset_name, 'resnet.pt'))

    return model

def ssl_encode(encoder, image):

    image.to(device)
    embedding = encoder(image_one=image, image_two=None)
    return embedding[0].detach()

def segment(img, seg_model, seg_hypar):
    img = np.array(img)
    image_tensor, orig_size = load_image(hypar=seg_hypar, img=img)
    mask = predict(seg_model,image_tensor,orig_size, seg_hypar, device)
    
    mask = Image.fromarray(mask)
    mask = mask.resize((image_dim,image_dim), Image.Resampling.LANCZOS)

    image = Image.fromarray(img)
    blank = image.point(lambda _: 0)
    obj = Image.composite(image, blank, mask)
    obj = object.resize((image_dim,image_dim), Image.Resampling.LANCZOS)

    return obj, mask

def inpaint(img, mask, inpaint_model):

    tuple = {}
    tuple['image'] = np.array(img)
    tuple['mask'] = np.array(mask)
    tuple['unpad_to_size'] = [torch.Tensor([256]), torch.Tensor([256])]
    batch = default_collate([tuple])

    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = inpaint_model(batch)                    
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

    bg = np.clip(cur_res * 255, 0, 255).astype('uint8')
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    window_name = 'image'
    cv2.imshow(window_name, bg)    

    return bg

def final_train(encoder, classifier, seg_model, seg_hypar, inpaint_model):

    bg_embeddings = []
    obj_embeddings = []
    labels = []

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        img, label = data
        obj, mask = segment(img, seg_model, seg_hypar)
        bg = inpaint(img, mask, inpaint_model)

        obj_rep = ssl_encode(encoder, obj) 
        bg_rep = ssl_encode(encoder, bg) 

        obj_embeddings.append(obj_rep)
        bg_embeddings.append(bg_rep)
        labels.append(label)

    bg_embeddings = torch.stack(bg_embeddings)
    conf_dict = reduce(bg_embeddings, n_clusters)
    conf_dict = torch.from_numpy(conf_dict)
    ccim = CCIM(1, image_dim, strategy='dp_cause')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr = lr, eps=1e-8)

    print("Training")
    classifier.train()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for i in range(len(bg_embeddings)):

            joint_rep = torch.cat((obj_embeddings[i], bg_embeddings[i]), 0).unsqueeze(1).cpu()
            final_rep = ccim(joint_rep, conf_dict).to(device)
            output = classifier(final_rep)
            loss = criterion(output, labels[i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += acc / len(bg_embeddings)
            epoch_loss += loss / len(bg_embeddings)

            print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")
    
    return classifier

def final_test(encoder, model, seg_model, seg_hypar, inpaint_model):

    bg_embeddings = []
    obj_embeddings = []
    labels = []

    y_pred = []
    y_true = []

    for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        img, label = data
        obj, mask = segment(img, seg_model, seg_hypar)
        bg = inpaint(img, mask, inpaint_model)

        obj_rep = ssl_encode(encoder, obj) 
        bg_rep = ssl_encode(encoder, bg) 

        obj_embeddings.append(obj_rep)
        bg_embeddings.append(bg_rep)
        labels.append(label)

    bg_embeddings = torch.stack(bg_embeddings)
    conf_dict = reduce(bg_embeddings, n_clusters)
    conf_dict = torch.from_numpy(conf_dict)
    ccim = CCIM(1, image_dim, strategy='dp_cause')

    print("Testing")

    model.eval()

    for i in range(len(bg_embeddings)):

        joint_rep = torch.cat((obj_embeddings[i], bg_embeddings[i]), 0).unsqueeze(1).cpu()
        final_rep = ccim(joint_rep, conf_dict).to(device)
        output = classifier(final_rep)

        _, prediction = output.data.max(1)
        y_pred.extend(prediction.tolist())
        y_true.extend(labels.tolist())

        print("Accuracy:{:.4f}".format(accuracy_score(y_true, y_pred) ))
        print("Recall:{:.4f}".format(recall_score(y_true, y_pred,average='macro') ))
        print("Precision:{:.4f}".format(precision_score(y_true, y_pred,average='macro') ))
        print("f1_score:{:.4f}".format(f1_score(y_true, y_pred,average='macro')))


def main():
    pt_resnet = models.resnet50(pretrained=True).to(device)

    ssl_encoder = train_ssl(pt_resnet, load_weights = True)
    print('BYOL Trained.')

    encoder = BYOL(
        ssl_encoder,
        image_size = 512,
        hidden_layer = 'avgpool',
        return_embedding = True
    )

    seg_model, seg_hypar = seg_model(1024, 42)

    # obj, mask = segment(img, net, hypar)
    inpaint_model = init_inpaint()

    classifier = classifier().to(device)
    classifier = final_train(encoder, classifier, seg_model, seg_hypar, inpaint_model)
    final_test(encoder, classifier, seg_model, seg_hypar, inpaint_model)

main()

    