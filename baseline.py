import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from configs import *
from PIL import Image
import timm
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
torch.backends.cudnn.enabled = False

# image_dir = os.path.join(root_dir, 'CIFAR-10')
# train_dir = os.path.join(image_dir, 'train')
# val_dir = os.path.join(image_dir, 'val')

epochs = 5
lr = 1e-3

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

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle = True, pin_memory=True, drop_last=True)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle = True, pin_memory=True, drop_last=True)

def classifier_train(classifier):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr = lr, eps=1e-8)
    
    for epoch in range(epochs):
        
        train_epoch_loss = 0
        train_epoch_accuracy = 0
        
        classifier.train()

        for (img, label) in tqdm(train_loader):
            optimizer.zero_grad()

            y_pred = []
            y_true = []

            output = classifier(img.to(device))
            loss = criterion(output, label.cuda())

            loss.backward()
            optimizer.step()

            _, preds = output.data.max(1)
            y_pred.extend(preds.tolist())
            y_true.extend(label.tolist())  

            acc = accuracy_score(y_true, y_pred)
            train_epoch_accuracy += acc / len(train_loader)
            train_epoch_loss += loss / len(train_loader)
            
        val_epoch_loss = 0
        val_epoch_accuracy = 0
        
        classifier.eval()

        for (img, label) in tqdm(val_loader):
            with torch.no_grad():
                y_pred = []
                y_true = []
                
                output = classifier(img.to(device))

                loss = criterion(output, label.cuda())
                _, preds = output.data.max(1)
                y_pred.extend(preds.tolist())
                y_true.extend(label.tolist())  

                acc = accuracy_score(y_true, y_pred)
                val_epoch_accuracy += acc / len(val_loader)
                val_epoch_loss += loss / len(val_loader)
            
        print(f"Epoch : {epoch+1} - train_loss : {train_epoch_loss:.4f} - train_acc: {train_epoch_accuracy:.4f} - val_loss : {val_epoch_loss:.4f} - val_acc: {val_epoch_accuracy:.4f}\n")
    
    torch.save(classifier.state_dict(),  os.path.join(weight_dir, dataset_name, 'classifier.pt'))

    return classifier

def classifier_test(classifier):

    y_pred = []
    y_true = []

    print("Testing")

    classifier.eval()

    for (img, label) in tqdm(val_loader):

        output = classifier(img.to(device))

        _, preds = output.data.max(1)
        y_pred.extend(preds.tolist())
        y_true.extend(label.tolist())  
                
    print(classification_report(y_true, y_pred))
    print(y_pred)

def main():
    classifier = timm.create_model('resnet50', pretrained=True, num_classes=9).to(device)
    classifier = classifier_train(classifier)
    classifier_test(classifier)

if __name__ == "__main__":
    main()