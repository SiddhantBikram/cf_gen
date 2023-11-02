import torch
from byol_loader import BYOL, TwoCropsTransform
from torchvision import models
import os
import numpy as np
from torchvision import transforms
import warnings
import torchvision.datasets as datasets
from tqdm import tqdm
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from torch.utils.data._utils.collate import default_collate
from sklearn.metrics import classification_report

from dis_loader import *
from backdoor_loader import *
from configs import *
from lama_loader import *
from dataloader import *
from cvae_loader import *
from utils import *

seed_everything(seed)
      
ssl_train_dataset = datasets.ImageFolder(train_dir, TwoCropsTransform(transforms.Compose(augmentation)))
ssl_train_loader = torch.utils.data.DataLoader(ssl_train_dataset, batch_size=16, shuffle = False, pin_memory=True, drop_last=True)
train_dataset = ImageFolderWithPaths(train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle = False, pin_memory=True, drop_last=True)
val_dataset = ImageFolderWithPaths(val_dir, transform=val_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle = False, pin_memory=True, drop_last=True)

class classifier_model(nn.Module):

    def __init__(self):
        super(classifier_model, self).__init__()

        self.obj = nn.Linear(image_dim, 128)
        self.bg = nn.Linear(image_dim, 128)
        self.fc = nn.Linear(image_dim, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, obj, bg):
       
        obj = self.dropout(F.relu(self.obj(obj)))
        bg = self.dropout(F.relu(self.bg(bg)))
        concat = torch.cat((obj, bg), 1)

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


def classifier_train(classifier, train_obj_embeddings, train_bg_embeddings, val_obj_embeddings, val_bg_embeddings, train_labels, val_labels):

    conf_dict = reduce(train_bg_embeddings, n_clusters)
    conf_dict = torch.from_numpy(conf_dict)
    train_ccim = CCIM(1, image_dim, strategy='dp_cause')

    val_conf_dict = reduce(val_bg_embeddings, n_clusters)
    val_conf_dict = torch.from_numpy(val_conf_dict)
    val_ccim = CCIM(1, image_dim, strategy='dp_cause')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr = lr, eps=1e-8)

    # for i in range(len(train_labels)):
    #     try:
    #         if train_labels[i] == 1:
    #             train_bg_embeddings = np.delete(train_bg_embeddings, 0)
    #             train_labels = np.delete(train_labels, 0)
    #             train_obj_embeddings = np.delete(train_obj_embeddings, 0)
    #     except:
    #         break

    train_obj_embeddings = torch.utils.data.Subset(train_obj_embeddings, range(len(train_obj_embeddings)-522))
    train_bg_embeddings = torch.utils.data.Subset(train_bg_embeddings, range(len(train_bg_embeddings)-522))
    train_labels = torch.utils.data.Subset(train_labels, range(len(train_labels)-522))


    print(len(train_labels))

    unique, counts = np.unique(train_labels, return_counts=True)

    print(np.asarray((unique, counts)).T)

    train_embed_dataset = custom_dataset(train_obj_embeddings, train_bg_embeddings, train_labels)
    train_embed_loader = torch.utils.data.DataLoader(train_embed_dataset, batch_size=embed_batch_size, shuffle = True, pin_memory=False, drop_last=True)

    val_embed_dataset = custom_dataset(val_obj_embeddings, val_bg_embeddings, val_labels)
    val_embed_loader = torch.utils.data.DataLoader(val_embed_dataset, batch_size=embed_batch_size, shuffle = True, pin_memory=False, drop_last=True)
    
    for epoch in range(epochs):
        
        train_epoch_loss = 0
        train_epoch_accuracy = 0
        
        classifier.train()

        for (obj, bg, label) in tqdm(train_embed_loader):
            optimizer.zero_grad()

            y_pred = []
            y_true = []
            
            # joint_rep = [torch.cat((obj[i], bg[i]), 0).to(device) for i in range(len(obj))]
            # joint_rep = torch.stack(joint_rep)
            # joint_rep = train_ccim(joint_rep, conf_dict).detach().squeeze(1).to(device)
            output = classifier(obj.to(device), bg.to(device))

            # joint_rep = torch.cat((obj[0], bg[0]), 0).to(device)


            loss = criterion(output, label.cuda())
            loss.backward()
            optimizer.step()

            _, preds = output.data.max(1)
            y_pred.extend(preds.tolist())
            y_true.extend(label.tolist())  

            acc = accuracy_score(y_true, y_pred)
            train_epoch_accuracy += acc / len(train_embed_loader)
            train_epoch_loss += loss / len(train_embed_loader)
            
        # print(f"Epoch : {epoch+1} - loss : {train_epoch_loss:.4f} - acc: {train_epoch_accuracy:.4f}\n")

        val_epoch_loss = 0
        val_epoch_accuracy = 0
        
        classifier.eval()

        for (obj, bg, label) in val_embed_loader:
            with torch.no_grad():

                y_pred = []
                y_true = []
                
                # joint_rep = [torch.cat((obj[i], bg[i]), 0).to(device) for i in range(len(obj))]
                # joint_rep = torch.stack(joint_rep)

                # joint_rep = val_ccim(joint_rep, val_conf_dict).detach().squeeze(1).to(device)
                output = classifier(obj.to(device), bg.to(device))

                # joint_rep = torch.cat((obj[0], bg[0]), 0).to(device)

                loss = criterion(output, label.cuda())
                _, preds = output.data.max(1)
                y_pred.extend(preds.tolist())
                y_true.extend(label.tolist())  

                acc = accuracy_score(y_true, y_pred)
                val_epoch_accuracy += acc / len(val_embed_loader)
                val_epoch_loss += loss / len(val_embed_loader)
            
        print(f"Epoch : {epoch+1} - train_loss : {train_epoch_loss:.4f} - train_acc: {train_epoch_accuracy:.4f} - val_loss : {val_epoch_loss:.4f} - val_acc: {val_epoch_accuracy:.4f}\n")
    
    torch.save(classifier.state_dict(),  os.path.join(weight_dir, dataset_name, 'classifier.pt'))

    return classifier

def classifier_test(classifier, val_obj_embeddings, val_bg_embeddings, val_labels):

    y_pred = []
    y_true = []

    conf_dict = reduce(val_bg_embeddings, n_clusters)
    conf_dict = torch.from_numpy(conf_dict)
    ccim = CCIM(1, image_dim, strategy='dp_cause')

    val_embed_dataset = custom_dataset(val_obj_embeddings, val_bg_embeddings, val_labels)
    val_embed_loader = torch.utils.data.DataLoader(val_embed_dataset, batch_size=embed_batch_size, shuffle = True, pin_memory=False, drop_last=True)

    print("Testing")

    classifier.eval()

    for (obj, bg, label) in tqdm(val_embed_loader):

        with torch.no_grad():

            joint_rep = [torch.cat((obj[i], bg[i]), 0).to(device) for i in range(len(obj))]
            joint_rep = torch.stack(joint_rep)

            # final_rep = ccim(joint_rep, conf_dict).detach().squeeze(1).to(device)
            output = classifier(obj.to(device), bg.to(device))

            _, preds = output.data.max(1)
            y_pred.extend(preds.tolist())
            y_true.extend(label.tolist())
                
    print(classification_report(y_true, y_pred))

def generate_samples(train_obj_embeddings, train_bg_embeddings, train_labels, n_classes):

    cvae = train_cvae(input_size=256, latent_size=20, n_epochs= 10, units = 200, train_obj_embeddings = train_obj_embeddings, train_labels= train_labels, n_classes = n_classes)

    head_class = []

    for i in range(len(train_obj_embeddings)):
        if train_labels[i] == 0:
            head_class.append(train_obj_embeddings[i])
            train_bg_embeddings = torch.cat((train_bg_embeddings, train_bg_embeddings[i].unsqueeze(0)),0)

    tail_class = [0,1]
    tail_class = torch.Tensor(tail_class)
    classes = [tail_class for i in range(len(head_class))]

    head_class = torch.stack(head_class)
    classes = torch.stack(classes)

    cvae.eval()

    mu, logvar = cvae.encoding_model(Variable((head_class)), classes)
    z = cvae.reparametrize(mu, logvar)
    generated_samples = cvae.decoding_model(z, classes)

    train_obj_embeddings= torch.cat((train_obj_embeddings, generated_samples), 0)
    new_labels= [1 for i in range(len(generated_samples))]
    train_labels = torch.cat((train_labels, torch.Tensor(new_labels)),0)

    torch.save(train_obj_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'aug_train_obj_embeddings.pt'))
    torch.save(train_bg_embeddings, os.path.join(root_dir, 'weights', dataset_name, 'aug_train_bg_embeddings.pt'))
    torch.save(train_labels, os.path.join(root_dir, 'weights', dataset_name, 'aug_train_labels.pt'))
    
    return train_obj_embeddings, train_bg_embeddings, train_labels

def main():

    load_embeddings = False
    load_weights = True

    if load_embeddings:
        train_obj_embeddings, train_bg_embeddings, val_obj_embeddings, val_bg_embeddings, train_labels, val_labels = load_embeddings_fn()
    
    else:
        pt_model = models.resnet50(pretrained=True).to(device)
        ssl_encoder = pt_model
        ssl_encoder = train_ssl(pt_model, load_weights = load_weights)
        print('BYOL Trained.')

        encoder = BYOL(
            ssl_encoder,
            image_size = image_dim,
            hidden_layer = 'avgpool',
            return_embedding = True
        )

        seg_model, seg_hypar = init_seg(1024, seed)

        inpaint_model = init_inpaint()

        train_obj_embeddings, train_bg_embeddings, val_obj_embeddings, val_bg_embeddings, train_labels, val_labels = make_embeddings(encoder, seg_model, seg_hypar, inpaint_model, train_loader, val_loader)

    # train_obj_embeddings, train_bg_embeddings, train_labels = generate_samples(train_obj_embeddings, train_bg_embeddings, train_labels, n_classes)
    # train_labels = [int(i) for i in train_labels]
    # train_obj_embeddings = [i.detach().cpu() for i in train_obj_embeddings]

    classifier = classifier_model().to(device)
    classifier = classifier_train(classifier, train_obj_embeddings, train_bg_embeddings, val_obj_embeddings, val_bg_embeddings, train_labels, val_labels)
    classifier_test(classifier, val_obj_embeddings, val_bg_embeddings, val_labels)
  
if __name__ == "__main__":
    main()

    