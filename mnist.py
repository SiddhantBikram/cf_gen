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
import psutil
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from torch.utils.data._utils.collate import default_collate
import cv2
from omegaconf import OmegaConf
import yaml
from dis_loader import *

from backdoor_loader import *
from configs import *
from dis_loader import *
from mnist_loader import *

epochs = 50
ssl_epochs = 5
lr = 1e-6
gamma = 0.7
seed = 510

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    g = torch.Generator()
    g.manual_seed(seed)
    
seed_everything(seed)

image_dim = 28

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

train_transforms = transforms.Compose(
    [
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor(),
    ]
)

class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
  
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = str(self.imgs[index][0])
        
        return (img, label ,path)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, objs, bgs, labels):
        self.objs = objs
        self.bgs = bgs
        self.labels = labels

    def __getitem__(self, idx):

        obj = self.objs[idx]
        bg = self.bgs[idx]
        label = self.labels[idx]

        return (obj, bg, label)

    def __len__(self):
        return len(self.labels)   
      

ssl_train_dataset = ColoredMNIST(root='/content/drive/MyDrive/Scripts', env='all_train')
ssl_train_loader = torch.utils.data.DataLoader(ssl_train_dataset, batch_size=16, shuffle = False, pin_memory=True, drop_last=True)
train_dataset = ColoredMNIST(root='/content/drive/MyDrive/Scripts', env='all_train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle = False, pin_memory=True, drop_last=True)
val_dataset = ColoredMNIST(root='/content/drive/MyDrive/Scripts', env='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle = False, pin_memory=True, drop_last=True)

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

def init_seg(size, seed):
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

class classifier_model(nn.Module):

    def __init__(self):
        super(classifier_model, self).__init__()

        # self.fc1 = nn.Linear(512, 128)
        # self.fc2 = nn.Linear(128, n_classes)
        self.fc = nn.Linear(image_dim*2, n_classes)

        # self.dropout = nn.Dropout(0.25)

    def forward(self, concat):
      
    #   linear_out = self.dropout(F.relu(self.fc1(concat)))    

    #   final_out = self.fc2(linear_out)   

        final_out = self.fc(concat)

        return final_out

def train_ssl(model, load_weights = True):

    print("Training BYOL")

    if load_weights:
        model.state_dict(torch.load(os.path.join(weight_dir, dataset_name, 'ssl_encoder.pt')))

        return model

    learner = BYOL(
        model,
        image_size = image_dim,
        hidden_layer = 'avgpool',
        return_embedding = False
    )

    opt = torch.optim.Adam(learner.parameters(), lr=1e-5)

    learner.train()

    for i in range(ssl_epochs):
        for j, (images, _) in enumerate(ssl_train_loader):

            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)

            loss = learner(image_one=images[0], image_two=images[1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() 

        print("Epoch: ",i," Loss: ", loss.item())
    torch.save(model.state_dict(),  os.path.join(weight_dir, dataset_name, 'ssl_encoder.pt'))

    return model

def ssl_encode(encoder, image):
    # print(image.shape)
    # image = np.swapaxes(image,0,2)
    # pil = Image.fromarray(image)
    # pil.show()
    encoder.eval()

    embedding = encoder(image_one=image, image_two=None)
    return embedding[0].detach()

def segment(img, path, seg_model, seg_hypar):

    image_tensor, orig_size = load_image(hypar=seg_hypar, im_path=path)
    mask = predict(seg_model,image_tensor,orig_size, seg_hypar, device)

    mask = Image.fromarray(mask)
    mask = mask.resize((image_dim,image_dim), Image.Resampling.LANCZOS)

    image = Image.open(path)
    image = image.resize((image_dim,image_dim), Image.Resampling.LANCZOS)
    blank = image.point(lambda _: 0)
    obj = Image.composite(image, blank, mask)
    obj = obj.resize((image_dim,image_dim), Image.Resampling.LANCZOS)

    return obj, mask

def inpaint(img, path, mask, inpaint_model):

    tuple = {}
    tuple['image'] = np.array(img)
    tuple['mask'] = np.expand_dims(np.array(mask), axis=0)
    tuple['unpad_to_size'] = [torch.Tensor([256]), torch.Tensor([256])]
    batch = default_collate([tuple])

    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = inpaint_model(batch)                    
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

    bg = np.clip(cur_res * 255, 0, 255).astype('uint8')
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    return bg

def classifier_train(encoder, classifier, seg_model, seg_hypar, inpaint_model, load_embeddings = True):

    if load_embeddings == True:
        train_bg_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'train_bg_embeddings.pt'))
        train_obj_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'train_obj_embeddings.pt'))
        train_labels = torch.load(os.path.join(weight_dir, dataset_name, 'train_labels.pt'))

    else:
        print('Making training embeddings')
        train_bg_embeddings = []
        train_obj_embeddings = []
        train_labels = []
        for (img, label, path) in tqdm(train_loader):
            obj, mask = segment(img[0], path[0], seg_model, seg_hypar)
            bg = inpaint(img[0], path[0], mask, inpaint_model)

            obj_rep = ssl_encode(encoder, transforms.ToTensor()(obj).unsqueeze_(0)) 
            bg_rep = ssl_encode(encoder, transforms.ToTensor()(bg).unsqueeze_(0)) 

            obj_embeddings.append(obj_rep.cpu().detach())
            bg_embeddings.append(bg_rep.cpu().detach())
            
            labels.append(label[0])

        bg_embeddings = torch.stack(bg_embeddings)
        obj_embeddings = torch.stack(obj_embeddings)
        labels = torch.stack(labels)
        
        torch.save(obj_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'train_obj_embeddings.pt'))
        torch.save(bg_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'train_bg_embeddings.pt'))
        torch.save(labels, os.path.join(root_dir, 'weights', dataset_name, 'train_labels.pt'))
      
    bg_embeddings = bg_embeddings.squeeze(1)
    obj_embeddings = obj_embeddings.squeeze(1)

    conf_dict = reduce(bg_embeddings, n_clusters)
    conf_dict = torch.from_numpy(conf_dict)
    ccim = CCIM(1, image_dim, strategy='dp_cause')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr = lr, eps=1e-8)

    train_embed_dataset = Dataset(obj_embeddings, bg_embeddings, labels)

    train_embed_loader = torch.utils.data.DataLoader(train_embed_dataset, batch_size=1, shuffle = True, pin_memory=False, drop_last=True)

    print("Training")

    classifier.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for (obj, bg, label) in tqdm(train_embed_loader):
            
            joint_rep = torch.cat((obj[0], bg[0]), 0).unsqueeze(1).cpu()

            final_rep = ccim(joint_rep, conf_dict).detach().squeeze(1).to(device)
            output = classifier(final_rep)

            loss = criterion(output, label[0].cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.cpu().argmax(dim=0) == label).float().mean()
            epoch_accuracy += acc / len(bg_embeddings)
            epoch_loss += loss / len(bg_embeddings)
            
        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")
    
    torch.save(classifier.state_dict(),  os.path.join(weight_dir, dataset_name, 'classifier.pt'))

    return classifier

def classifier_test(encoder, classifier, seg_model, seg_hypar, inpaint_model, load_embeddings = True):

    bg_embeddings = []
    obj_embeddings = []
    labels = []

    y_pred = []
    y_true = []


    if load_embeddings == True:
        bg_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'val_bg_embeddings.pt'))
        obj_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'val_obj_embeddings.pt'))
        labels = torch.load(os.path.join(weight_dir, dataset_name, 'val_labels.pt'))

    else:
    
        print('Making test embeddings')

        for (img, label, path) in tqdm(val_loader):
            obj, mask = segment(img[0], path[0], seg_model, seg_hypar)
            bg = inpaint(img[0], path[0], mask, inpaint_model)

            obj_rep = ssl_encode(encoder, transforms.ToTensor()(obj).unsqueeze_(0)) 
            bg_rep = ssl_encode(encoder, transforms.ToTensor()(bg).unsqueeze_(0)) 

            obj_embeddings.append(obj_rep.cpu().detach())
            bg_embeddings.append(bg_rep.cpu().detach())
            labels.append(label[0])

        bg_embeddings = torch.stack(bg_embeddings)
        obj_embeddings = torch.stack(obj_embeddings)
        labels = torch.stack(labels)

        torch.save(obj_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'val_obj_embeddings.pt'))
        torch.save(bg_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'val_bg_embeddings.pt'))
        torch.save(labels, os.path.join(root_dir, 'weights', dataset_name, 'val_labels.pt'))

    bg_embeddings = bg_embeddings.squeeze(1)
    obj_embeddings = obj_embeddings.squeeze(1)

    conf_dict = reduce(bg_embeddings, n_clusters)
    conf_dict = torch.from_numpy(conf_dict)
    ccim = CCIM(1, image_dim, strategy='dp_cause')

    val_embed_dataset = Dataset(obj_embeddings, bg_embeddings, labels)

    val_embed_loader = torch.utils.data.DataLoader(val_embed_dataset, batch_size=1, shuffle = True, pin_memory=False, drop_last=True)

    print("Testing")

    classifier.eval()

    for (obj, bg, label) in tqdm(val_embed_loader):

        joint_rep = torch.cat((obj[0], bg[0]), 0).unsqueeze(1).cpu()
        final_rep = ccim(joint_rep, conf_dict).detach().squeeze(1).to(device)
        output = classifier(final_rep)

        prediction = output.cpu().argmax(dim=0)
        y_pred.append(prediction.item())
        y_true.append(label[0])
                
    print("Accuracy:{:.4f}".format(accuracy_score(y_true, y_pred) ))
    print("Recall:{:.4f}".format(recall_score(y_true, y_pred,average='macro') ))
    print("Precision:{:.4f}".format(precision_score(y_true, y_pred,average='macro') ))
    print("f1_score:{:.4f}".format(f1_score(y_true, y_pred,average='macro')))

    print(y_pred)

def train_val_test(encoder, classifier, seg_model, seg_hypar, inpaint_model, load_embeddings = True):

    y_pred = []
    y_true = []

    if load_embeddings == True:
        train_bg_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'train_bg_embeddings.pt'))
        train_obj_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'train_obj_embeddings.pt'))
        train_labels = torch.load(os.path.join(weight_dir, dataset_name, 'train_labels.pt'))

        val_bg_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'val_bg_embeddings.pt'))
        val_obj_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'val_obj_embeddings.pt'))
        val_labels = torch.load(os.path.join(weight_dir, dataset_name, 'val_labels.pt'))

    else:
        print('Making training embeddings')

        train_bg_embeddings = []
        train_obj_embeddings = []
        train_labels = []

        for (img, label, path) in tqdm(train_loader):
            # obj, mask = segment(img[0], path[0], seg_model, seg_hypar)
            # bg = inpaint(img[0], path[0], mask, inpaint_model)

            obj_rep = ssl_encode(encoder, transforms.ToTensor()(obj).unsqueeze_(0)) 
            bg_rep = ssl_encode(encoder, transforms.ToTensor()(bg).unsqueeze_(0)) 

            train_obj_embeddings.append(obj_rep.cpu().detach())
            train_bg_embeddings.append(bg_rep.cpu().detach())
            train_labels.append(label[0])

        train_bg_embeddings = torch.stack(train_bg_embeddings)
        train_obj_embeddings = torch.stack(train_obj_embeddings)
        train_labels = torch.stack(train_labels)
        
        torch.save(train_obj_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'train_obj_embeddings.pt'))
        torch.save(train_bg_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'train_bg_embeddings.pt'))
        torch.save(train_labels, os.path.join(root_dir, 'weights', dataset_name, 'train_labels.pt'))
        
        val_bg_embeddings = []
        val_obj_embeddings = []
        val_labels = []

        print('Making test embeddings')

        for (img, label, path) in tqdm(val_loader):
            obj, mask = segment(img[0], path[0], seg_model, seg_hypar)
            bg = inpaint(img[0], path[0], mask, inpaint_model)

            obj_rep = ssl_encode(encoder, transforms.ToTensor()(obj).unsqueeze_(0)) 
            bg_rep = ssl_encode(encoder, transforms.ToTensor()(bg).unsqueeze_(0)) 

            val_obj_embeddings.append(obj_rep.cpu().detach())
            val_bg_embeddings.append(bg_rep.cpu().detach())
            val_labels.append(label[0])

        val_bg_embeddings = torch.stack(val_bg_embeddings)
        val_obj_embeddings = torch.stack(val_obj_embeddings)
        val_labels = torch.stack(val_labels)

        torch.save(val_obj_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'val_obj_embeddings.pt'))
        torch.save(val_bg_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'val_bg_embeddings.pt'))
        torch.save(val_labels, os.path.join(root_dir, 'weights', dataset_name, 'val_labels.pt'))

    val_bg_embeddings = val_bg_embeddings.squeeze(1)
    val_obj_embeddings = val_obj_embeddings.squeeze(1)
    train_bg_embeddings = train_bg_embeddings.squeeze(1)
    train_obj_embeddings = train_obj_embeddings.squeeze(1)

    conf_dict = reduce(train_bg_embeddings, n_clusters)
    conf_dict = torch.from_numpy(conf_dict)
    ccim = CCIM(1, image_dim, strategy='dp_cause')

    val_conf_dict = reduce(val_bg_embeddings, n_clusters)
    val_conf_dict = torch.from_numpy(val_conf_dict)
    val_ccim = CCIM(1, image_dim, strategy='dp_cause')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr = lr, eps=1e-8)

    train_embed_dataset = Dataset(train_obj_embeddings, train_bg_embeddings, train_labels)
    train_embed_loader = torch.utils.data.DataLoader(train_embed_dataset, batch_size=1, shuffle = True, pin_memory=False, drop_last=True)

    val_embed_dataset = Dataset(val_obj_embeddings, val_bg_embeddings, val_labels)
    val_embed_loader = torch.utils.data.DataLoader(val_embed_dataset, batch_size=1, shuffle = True, pin_memory=False, drop_last=True)

    
    for epoch in range(epochs):
        
        train_epoch_loss = 0
        train_epoch_accuracy = 0
        
        classifier.train()

        for (obj, bg, label) in tqdm(train_embed_loader):
            
            # joint_rep = torch.cat((obj[0], bg[0]), 0).unsqueeze(1).cpu()
            joint_rep = torch.cat((obj[0], bg[0]), 0).to(device)
            # final_rep = ccim(joint_rep, conf_dict).detach().squeeze(1).to(device)
            # output = classifier(final_rep)
            output = classifier(joint_rep)

            loss = criterion(output, label[0].cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.cpu().argmax(dim=0) == label).float().mean()
            train_epoch_accuracy += acc / len(train_bg_embeddings)
            train_epoch_loss += loss / len(train_bg_embeddings)
            
        # print(f"Epoch : {epoch+1} - loss : {train_epoch_loss:.4f} - acc: {train_epoch_accuracy:.4f}\n")

        val_epoch_loss = 0
        val_epoch_accuracy = 0
        
        classifier.eval()

        for (obj, bg, label) in tqdm(val_embed_loader):
            joint_rep = torch.cat((obj[0], bg[0]), 0).to(device)
            # joint_rep = torch.cat((obj[0], bg[0]), 0).unsqueeze(1).cpu()

            # final_rep = val_ccim(joint_rep, val_conf_dict).detach().squeeze(1).to(device)
            # output = classifier(final_rep)
            output = classifier(joint_rep)

            loss = criterion(output, label[0].cuda())
            acc = (output.cpu().argmax(dim=0) == label).float().mean()
            val_epoch_accuracy += acc / len(val_bg_embeddings)
            val_epoch_loss += loss / len(val_bg_embeddings)
            
        print(f"Epoch : {epoch+1} - train_loss : {train_epoch_loss:.4f} - train_acc: {train_epoch_accuracy:.4f} - val_loss : {val_epoch_loss:.4f} - val_acc: {val_epoch_accuracy:.4f}\n")
    
    torch.save(classifier.state_dict(),  os.path.join(weight_dir, dataset_name, 'classifier.pt'))
    
    print("Testing")

    classifier.eval()

    for (obj, bg, label) in tqdm(val_embed_loader):
        joint_rep = torch.cat((obj[0], bg[0]), 0).to(device)
        # joint_rep = torch.cat((obj[0], bg[0]), 0).unsqueeze(1).cpu()
        # final_rep = ccim(joint_rep, val_conf_dict).detach().squeeze(1).to(device)
        # output = classifier(final_rep)
        output = classifier(joint_rep)

        prediction = output.cpu().argmax(dim=0)
        y_pred.append(prediction.item())
        y_true.append(label[0])
                
    print("Accuracy:{:.4f}".format(accuracy_score(y_true, y_pred) ))
    print("Recall:{:.4f}".format(recall_score(y_true, y_pred,average='macro') ))
    print("Precision:{:.4f}".format(precision_score(y_true, y_pred,average='macro') ))
    print("f1_score:{:.4f}".format(f1_score(y_true, y_pred,average='macro')))

    print(y_pred)


def main():
    pt_model = models.resnet50(pretrained=True).to(device)
    ssl_encoder = pt_model
    ssl_encoder = train_ssl(pt_model, load_weights = False)
    print('BYOL Trained.')

    encoder = BYOL(
        ssl_encoder,
        image_size = 512,
        hidden_layer = 'avgpool',
        return_embedding = True
    )

    seg_model, seg_hypar = init_seg(1024, 42)

    inpaint_model = init_inpaint()

    classifier = classifier_model().to(device)

    # classifier = classifier_train(encoder, classifier, seg_model, seg_hypar, inpaint_model, load_embeddings = True)
    # classifier.state_dict(torch.load(os.path.join(weight_dir, dataset_name, 'classifier.pt')))

    # classifier_test(encoder, classifier, seg_model, seg_hypar, inpaint_model, load_embeddings = True)

    train_val_test(encoder, classifier, seg_model, seg_hypar, inpaint_model, load_embeddings = False)
main()

    