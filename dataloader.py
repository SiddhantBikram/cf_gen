import torch
import os 
from torchvision import transforms
import torchvision.datasets as datasets
from byol_loader import GaussianBlur

from dis_loader import *
from configs import *
from lama_loader import *
from utils import *

augmentation = [
            transforms.RandomResizedCrop(image_dim, scale=(0.2, 1.)),
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
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((image_dim, image_dim)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)

class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
  
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = str(self.imgs[index][0])
        
        return (img, label ,path)

class custom_dataset(torch.utils.data.Dataset):

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
    
class gen_dataset(torch.utils.data.Dataset):

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):

        embedding = self.embeddings[idx]
        label = self.labels[idx]

        return (embedding, label)

    def __len__(self):
        return len(self.labels) 

def load_embeddings_fn():
    
    train_bg_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'train_bg_embeddings.pt'))
    train_obj_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'train_obj_embeddings.pt'))
    train_labels = torch.load(os.path.join(weight_dir, dataset_name, 'train_labels.pt'))

    val_bg_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'val_bg_embeddings.pt'))
    val_obj_embeddings = torch.load(os.path.join(weight_dir, dataset_name, 'val_obj_embeddings.pt'))
    val_labels = torch.load(os.path.join(weight_dir, dataset_name, 'val_labels.pt'))

    val_bg_embeddings = val_bg_embeddings.squeeze(1)
    val_obj_embeddings = val_obj_embeddings.squeeze(1)
    train_bg_embeddings = train_bg_embeddings.squeeze(1)
    train_obj_embeddings = train_obj_embeddings.squeeze(1)

    return train_obj_embeddings, train_bg_embeddings, val_obj_embeddings, val_bg_embeddings, train_labels, val_labels

def make_embeddings(encoder, seg_model, seg_hypar, inpaint_model, train_loader, val_loader):

    print('Making training embeddings')

    train_bg_embeddings = []
    train_obj_embeddings = []
    train_labels = []

    for (img, label, path) in tqdm(train_loader):
        obj, mask = segment(path[0], seg_model, seg_hypar)
        bg = inpaint(img[0], mask, inpaint_model)

        print(np.array(mask).shape)
        print(np.array(img[0]).shape)

        obj = Image.open(path[0])
        bg = Image.new(mode="RGB", size=(image_dim, image_dim))

        y_nonzero, x_nonzero, _ = np.nonzero(obj)
        obj = obj.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))
        obj= obj.resize((image_dim,image_dim))
        
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
        # obj, mask = segment(path[0], seg_model, seg_hypar)
        # bg = inpaint(img[0], mask, inpaint_model)

        obj = Image.open(path[0])
        bg = Image.new(mode="RGB", size=(image_dim, image_dim))

        y_nonzero, x_nonzero, _ = np.nonzero(obj)
        obj = obj.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))
        obj= obj.resize((image_dim,image_dim))

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

    return train_obj_embeddings, train_bg_embeddings, val_obj_embeddings, val_bg_embeddings, train_labels, val_labels