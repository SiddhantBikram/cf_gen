import torch 
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

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

class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, class_size, units):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.units = units
        self.encode1 = nn.Linear(input_size + self.class_size, self.units)
        self.encode2 = nn.Linear(self.units, self.units//2)
        self.encode3 = nn.Linear(self.units//2, latent_size)
        self.encode4 = nn.Linear(self.units//2, latent_size)
        self.decode1 = nn.Linear(latent_size + self.class_size, self.units//2)
        self.decode2 = nn.Linear(self.units//2, self.units)
        self.decode3 = nn.Linear(self.units, self.input_size)

    def encoding_model(self, x, c):
        theinput = torch.cat((x.float(), c.float()), 1)
        output = self.encode1(theinput)
        output = self.encode2(output)
        mu = self.encode3(output)
        logvar = self.encode4(output)
        return mu, logvar

    def decoding_model(self, z, c):
        z_input = torch.cat((z.float(), c.float()), 1)
        output = self.decode1(z_input)
        output = self.decode2(output)
        x_hat = self.decode3(output)
        return x_hat

    def forward(self, x, c):

        mu, logvar = self.encoding_model(x, c)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoding_model(z, c)
        return x_hat, mu, logvar
   
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        epsilon = Variable(std.data.new(std.size()).normal_())
        return epsilon.mul(std) + mu

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return Variable(targets)

def loss_function(x_hat, x, mu, logvar):
    reconstruction_function = nn.BCEWithLogitsLoss()
    reconstruction_function.size_average = True
    reconstruction_loss = reconstruction_function(x_hat, x)
    kl_divergence = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=1)
    kl_divergence = torch.sum(kl_divergence, dim=0)
    loss = (reconstruction_loss + kl_divergence)
    return loss

def train_cvae(input_size, latent_size, n_epochs, units, train_obj_embeddings, train_labels, n_classes):

    cvae = CVAE(input_size, latent_size, n_classes, units)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=0.001)

    dataset = gen_dataset(train_obj_embeddings, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle = True, pin_memory=False, drop_last=True)

    cvae.train()

    for epoch in range(n_epochs):
        for (embedding, labels) in train_loader:
            optimizer.zero_grad()
            labels = one_hot(labels, n_classes)
            reconstruction_batch, mu, logvar = cvae(embedding, labels)
            loss = loss_function(reconstruction_batch, embedding, mu, logvar)
            loss.backward()
            optimizer.step()
    
    return cvae

def test_cvae():

    train_obj_embeddings = torch.rand(100, 256)
    train_labels = []

    for i in range(len(train_obj_embeddings)):

        if i%2 ==0:
            train_labels.append(0)

            train_labels.append(1)

    cvae = train_cvae(input_size=256, latent_size=20, n_epochs= 10, units = 200, train_obj_embeddings = train_obj_embeddings, train_labels= train_labels)

    head_class = []

    for i in range(len(train_obj_embeddings)):
        if train_labels[i] == 0:
            head_class.append(train_obj_embeddings[i])

    tail_class = [0,1]
    tail_class = torch.tensor(tail_class)
    classes = [tail_class for i in range(len(head_class))]

    head_class = torch.stack(head_class)
    classes = torch.stack(classes)

    mu, logvar = cvae.encoding_model(Variable((head_class)), classes)
    z = cvae.reparametrize(mu, logvar)
    generated_samples = cvae.decoding_model(z, classes)

    train_obj_embeddings= torch.cat((train_obj_embeddings, generated_samples), 0)
    new_labels= [1 for i in range(len(generated_samples))]
    train_labels.extend(new_labels)
