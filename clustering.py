from sklearn.cluster import KMeans
import seaborn as sns
import torch
from sklearn.metrics import pairwise_distances_argmin_min
import torch.nn as nn

def clusters(data, point):

    kmeans = KMeans(n_clusters = 10, random_state = 0, init = 'k-means++').fit(data)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
    prediction = kmeans.predict(point.reshape(1, -1))

    centers = []

    for i in range(len(closest)):
        if kmeans.predict(data[closest[i]].reshape(1,-1)) != prediction[0]:
            centers.append(data[closest[i]])

    return kmeans, centers

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Linear(512, 256)
        self.decoder = nn.Linear(256, 256)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.encoder(x))
        x = self.decoder(x)

        return x


def main():

    point = torch.rand(1, 256)
    data = torch.rand(15, 256)
    kmeans, centers = clusters(data, point)
    model = autoencoder()
    loss = nn.CrossEntropyLoss()

    model.train()

    for point in data:
        for center_idx in range(len(centers)):
            new = model(torch.cat([point, centers[center_idx]]))
            distances = [torch.cdist(new.unsqueeze(0), i.unsqueeze(0))**2 for i in centers]
            inverse_dist = sum([1/i for i in distances])
            probabilities = [(1/dist)/inverse_dist for dist in distances]
            target = torch.zeros_like(centers)
            target[center_idx] = 1
            
            loss1 = loss(probabilities, target)
            

if __name__ == '__main__':
    main()